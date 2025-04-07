import os
import json

import yaml
import pickle
import numpy as np
import scipy.ndimage
from torch.utils.data import Dataset
from copy import deepcopy
from utils import read_config_and_load_components

class Geometry(object):
    def __init__(self, config):
        self.v_res = config['nVoxel'][0]  # ct scan
        self.p_res = config['nDetector'][0]  # projections
        self.v_spacing = np.array(config['dVoxel'])[0]  # mm
        self.p_spacing = np.array(config['dDetector'])[0]  # mm

        self.DSO = config['DSO']  # mm
        self.DSD = config['DSD']  # mm

    def project(self, points, angle):
        # points: [N, 3] ranging from [0, 1]
        # d_points: [N, 2] ranging from [-1, 1]

        points = deepcopy(points).astype(float)
        points[:, :2] -= 0.5  # [-0.5, 0.5]
        points[:, 2] = 0.5 - points[:, 2]  # [-0.5, 0.5]
        points *= self.v_res * self.v_spacing  # mm

        angle = -1 * angle  # inverse direction
        rot_M = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
        points = points @ rot_M.T

        d1 = self.DSO
        d2 = self.DSD

        coeff = (d2) / (d1 - points[:, 0])  # N,
        d_points = points[:, [2, 1]] * coeff[:, None]  # [N, 2] float
        d_points /= (self.p_res * self.p_spacing)
        d_points *= 2  # NOTE: some points may fall outside [-1, 1]

        return d_points

class Pelvis_Bone_Recon_Dataset(Dataset):

    def __init__(
            self,
            part = 'hip',
            data_root= None,
            split='train',
            dataset_config= 'hip_info.json',
            npoint=10000,
            out_res = 128,
            sdf_sample_range = 2,
            balance_ratio = 0.5,
            is_train = None
    ):
        super(Dataset, self).__init__()

        part_dict = {
            'hip': {
                'part': 'hip',
                'sub_part': ['hip_left', 'hip_right'],
            }
        }

        self.part_info = part_dict[part]

        # load dataset info
        with open(os.path.join(data_root, dataset_config), 'r') as f:
            cfg = json.load(f)
            self.cfg = cfg
            self.subject_list = sorted(cfg[split])

        # load projection config
        with open(os.path.join(data_root, cfg['drr_config']), 'r') as f:
            drr_cfg = yaml.safe_load(f)
            self.geo = Geometry(drr_cfg)

        # load full config
        full_cfg = read_config_and_load_components(os.path.join(data_root, 'TotalSegmentor-hips-DRR-full.yaml'))

        if is_train is None:
            self.is_train = (split == 'train')
        else:
            self.is_train = is_train

        # Prepare points
        if self.is_train:
            # load blocks' coordinates [train only]
            self.blocks = np.load(os.path.join(data_root, cfg['blocks']))
            self.sdf_sample_range = sdf_sample_range
            self.balance_ratio = balance_ratio

        self.out_res = out_res
        self.data_root = data_root
        self.npoint = npoint

        self.spatial_division_num = full_cfg["ROI_properties"]["spatial_division_num"]

        # prepare sampling points
        if not self.is_train:
            points = np.mgrid[:out_res, :out_res, :out_res]
            points = points.astype(float) / (out_res - 1)
            points = points.reshape(3, -1)
            self.points = points.transpose(1, 0)  # N, 3

            boundary_index = [self.spatial_division_num // 2 - i - 0.5 for i in range(self.spatial_division_num)]
            # Additional divide is included in the first and last subspace
            boundary_index[0] += 0.5
            boundary_index[-1] -= 0.5

            points = points * full_cfg["ROI_properties"]["size"] + 0.5
            center = np.array([full_cfg["ROI_properties"]["size"] / 2, full_cfg["ROI_properties"]["size"] / 2, full_cfg["ROI_properties"]["size"] / 2])
            points = points - center[:, None]

            angles = np.linspace(
                0,
                np.pi,
                3
            )[:-1] + drr_cfg["startAngle"] / 180 * np.pi

            multiview_continuous_weight = []
            for angle in angles:
                view_continuous_weight = np.zeros((self.spatial_division_num, out_res * out_res * out_res))
                assert angle <= np.pi
                if angle < np.pi / 4:
                    intersection_length = 336 / np.cos(angle)
                elif angle < np.pi / 4 * 2:
                    intersection_length = 336 / np.cos(np.pi / 2 - angle)
                elif angle < np.pi / 4 * 3:
                    intersection_length = 336 / np.cos(angle - np.pi / 2)
                else:
                    intersection_length = 336 / np.cos(np.pi - angle)

                view_direction_index = (points[0, ...] * np.cos(angle) + points[1, ...] * np.sin(angle)) / (intersection_length / (self.spatial_division_num + 2))
                for i in range(self.spatial_division_num):
                    view_continuous_weight[i] = 1 / ((view_direction_index - boundary_index[i]) ** 2 + 1e-3)
                view_continuous_weight = view_continuous_weight / np.sum(view_continuous_weight, axis=0, keepdims=True)
                multiview_continuous_weight.append(view_continuous_weight)
            self.points_complete_spatial_division_weight = np.stack(multiview_continuous_weight, axis=0).astype(np.float32)


    def __len__(self):
        return len(self.subject_list)

    def sample_points(self, points, of_values=None, sdf_values=None, spatial_idx = None, balance_ratio = 0.5, sdf_sample_range= 2):

        near_shape_in_mesh_points_idx = np.where(((sdf_values < 0) & (sdf_values >= -sdf_sample_range)))[0]
        uniform_in_mesh_points_idx = np.where(sdf_values < sdf_sample_range)[0]
        near_shape_out_mesh_points_idx = np.where(((sdf_values > 0) & (sdf_values <= sdf_sample_range)))[0]
        uniform_out_mesh_points_idx = np.where(sdf_values > sdf_sample_range)[0]

        in_mesh_points_num = int(self.npoint * 0.5)
        out_mesh_points_num = self.npoint - in_mesh_points_num

        near_shape_out_mesh_points_num = int(out_mesh_points_num * balance_ratio)
        if near_shape_out_mesh_points_num > len(near_shape_out_mesh_points_idx):
            near_shape_out_mesh_points_num = len(near_shape_out_mesh_points_idx)
        uniform_out_mesh_points_num = out_mesh_points_num - near_shape_out_mesh_points_num

        near_shape_in_mesh_points_num = int(in_mesh_points_num * balance_ratio)
        if near_shape_in_mesh_points_num > len(near_shape_in_mesh_points_idx):
            near_shape_in_mesh_points_num = len(near_shape_in_mesh_points_idx)
        uniform_in_mesh_points_num = in_mesh_points_num - near_shape_in_mesh_points_num

        near_shape_in_mesh_points_choices = np.random.choice(near_shape_in_mesh_points_idx, size=near_shape_in_mesh_points_num, replace=False)
        uniform_in_mesh_points_choices = np.random.choice(uniform_in_mesh_points_idx, size=uniform_in_mesh_points_num, replace=False)
        near_shape_out_mesh_points_choices = np.random.choice(near_shape_out_mesh_points_idx, size=near_shape_out_mesh_points_num, replace=False)
        uniform_out_mesh_points_choices = np.random.choice(uniform_out_mesh_points_idx, size=uniform_out_mesh_points_num, replace=False)

        choices = np.concatenate(
            (near_shape_in_mesh_points_choices,uniform_in_mesh_points_choices , near_shape_out_mesh_points_choices, uniform_out_mesh_points_choices))
        choices = np.sort(choices)

        points = points[choices]
        of_values = of_values[:, choices].astype(float)
        spatial_idx = spatial_idx[:, : , choices]

        return points, of_values, spatial_idx

    def load_spatial_division_drr(self, subject_id):

        with open(os.path.join(self.data_root, self.cfg[f'spatial_division_drr'].format(subject_id, subject_id)),'rb') as f:
            data = pickle.load(f)
            proj_infos = data['projections'].astype(float) / 255.  # M, K, W, H
            proj_infos = proj_infos[:, :, None, ...]  # M, K, 1, W, H

        return proj_infos

    def load_original_drr(self, subject_id):
        # -- load projections
        with open(os.path.join(self.data_root, self.cfg[f'original_drr'].format(subject_id, subject_id)), 'rb') as f:
            data = pickle.load(f)
            projs = data['projections']  # M, W, H
            angles = data['angles']  # M,
            projs = projs[:, None, ...].astype(float) / 255. # M, 1, W, H

        return projs, angles

    def load_occupancy_field(self, subject_id, sub_part):
        occupancy_field = np.load(os.path.join(self.data_root, self.cfg[f'{sub_part}_occupancy_field'].format(subject_id, subject_id)))
        if self.out_res != occupancy_field.shape[0]:
            occupancy_field = scipy.ndimage.zoom(
                occupancy_field,
                self.out_res / occupancy_field.shape[0],
                prefilter= False,
                order=0
            )
        return occupancy_field

    def load_of_block(self, subject_id, b_idx, sub_part):
        path = os.path.join(self.data_root, self.cfg[f'{sub_part}_of_block'].format(subject_id, b_idx))
        return np.load(path)

    def load_sdf_block(self, subject_id, b_idx):
        path = os.path.join(self.data_root, self.cfg['sdf_block'].format(subject_id, b_idx))
        return np.load(path)

    def load_spatial_weight_block(self, b_idx):
        path = os.path.join(self.data_root, self.cfg[f'inter_space_fusion_weight_blocks'].format(b_idx))
        return np.load(path)

    def __getitem__(self, index):
        subject_id = self.subject_list[index]

        # -- load projections
        original_drr, angles = self.load_original_drr(subject_id)

        divide_mask_drr = self.load_spatial_division_drr(subject_id)
        drrs = np.concatenate([original_drr[:, None, ...], divide_mask_drr], axis= 1) # M, 1 + K, 1, W, H

        # -- load sampling points
        if not self.is_train:
            points = self.points
            spatial_weight = self.points_complete_spatial_division_weight
            occupancy_field = []
            for sub_part in self.part_info['sub_part']:
                sub_occupancy_field = self.load_occupancy_field(subject_id, sub_part)
                occupancy_field.append(sub_occupancy_field[None, ...])
            occupancy_field = np.concatenate(occupancy_field, axis= 0)
            p_gt = np.zeros(len(points))
        else:
            b_idx = np.random.randint(len(self.blocks))
            of_block_values = []
            for sub_part in self.part_info['sub_part']:
                sub_of_block_values = self.load_of_block(subject_id, b_idx, sub_part)
                of_block_values.append(sub_of_block_values)
            of_block_values = np.stack(of_block_values, axis= 0)
            sdf_block_values = self.load_sdf_block(subject_id, b_idx)
            block_coords = self.blocks[b_idx]  # N, 3
            spatial_weight_values = self.load_spatial_weight_block(b_idx)
            points, p_gt, spatial_weight = self.sample_points(block_coords, of_block_values, sdf_block_values, spatial_weight_values, self.balance_ratio, self.sdf_sample_range)

        # -- project points and view direction
        proj_points = []
        for a in angles:
            p = self.geo.project(points, a)
            proj_points.append(p)
        proj_points = np.stack(proj_points, axis=0)  # M, N, 2


        mesh_gt_path = {}
        for sub_part in self.part_info['sub_part']:
            mesh_gt_path[sub_part] = os.path.join(self.data_root, self.cfg[f'{sub_part}_mesh_path'].format(subject_id, subject_id))

        # -- collect data
        ret_dict = {
            'subject_id': subject_id,
            'points': points,  # 3D points
            'proj_points': proj_points,  # projected points
            'drrs': drrs,  # 2D projections,
            'p_gt': p_gt,  # labels
            'mesh_gt_path': mesh_gt_path, # trimesh objects,
            'spatial_weight': spatial_weight
        }
        if not self.is_train:
            ret_dict['occupancy_field'] = occupancy_field # occupancy field serves as the segmentation label at the desired evaluation size

        return ret_dict