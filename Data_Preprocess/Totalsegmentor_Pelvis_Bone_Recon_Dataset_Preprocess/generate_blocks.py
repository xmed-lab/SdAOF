'''
Original sdf & of represents physical volume of (336, 336, 336) with spacing (1, 1, 1).
Loading and sampling from 336^3 points can be time-consuming, divide them into 4^3 * 84 * 84 * 84 blocks
'''

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)

import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial
from tqdm import tqdm
from Data_Preprocess.XrayTo3DPreprocess.xrayto3d_preprocess_utils import read_config_and_load_components


def generate_blocks():
    block_list = []
    partial_size = config["ROI_properties"]["size"] // 4
    base = np.mgrid[:partial_size, :partial_size, :partial_size] * 4  # 3, 84 ^ 3
    base = base.reshape(3, -1)
    for x in range(4):
        for y in range(4):
            for z in range(4):
                offset = np.array([x, y, z])
                block = base + offset[:, None]
                block_list.append(block)
    return block_list

def generate_filed_blocks(subject_id, field_type, block_list, mesh_dir, field_blocks_dir):
    field_file = os.path.join(mesh_dir, f'{subject_id}/{subject_id}_{field_type}.npy')
    field = np.load(field_file)

    # save_dir = f'./field_blocks/{subject_id}/'
    save_dir = os.path.join(field_blocks_dir, subject_id)
    os.makedirs(save_dir, exist_ok=True)
    for k, block in enumerate(block_list):
        block = block.reshape(3, -1).transpose(1, 0)
        field_block = field[block[:, 0], block[:, 1], block[:, 2]]
        np.save(os.path.join(save_dir, f'{field_type}_block_{k}.npy'), field_block)

def generate_inter_space_fusion_weights_blocks():
    spatial_size = config["ROI_properties"]["size"]
    divide_num = config["ROI_properties"]["spatial_division_num"]
    view = config["CT_TIGRE_config"]["numTrain"]
    AngleOffset = config["CT_TIGRE_config"]["startAngle"]

    block_list = []
    boundary_index = [divide_num // 2 - i - 0.5 for i in range(divide_num)]
    # Additional divide is included in the first and last subspace
    boundary_index[0] += 0.5
    boundary_index[-1] -= 0.5

    points = np.mgrid[:spatial_size // 4, :spatial_size // 4, :spatial_size // 4] * 4 # 3, 84 ^ 3
    points = points.reshape(3, -1).astype(float)
    center = np.array([spatial_size // 2, spatial_size // 2, spatial_size // 2])

    angles = np.linspace(
        0,
        np.pi,
        view + 1
    )[:-1] + AngleOffset / 180 * np.pi

    for x in range(4):
        for y in range(4):
            for z in range(4):
                multiview_continuous_weight = []
                offset = np.array([x, y, z])
                block = points + offset[:, None] - center[..., None] + 0.5
                for angle in angles:
                    view_continuous_weight = np.zeros((divide_num, spatial_size // 4 * spatial_size // 4 * spatial_size // 4))
                    assert angle <= np.pi
                    if angle < np.pi / 4:
                        intersection_length = spatial_size / np.cos(angle)
                    elif angle < np.pi / 4 * 2:
                        intersection_length = spatial_size / np.cos(np.pi / 2 - angle)
                    elif angle < np.pi / 4 * 3:
                        intersection_length = spatial_size / np.cos(angle - np.pi / 2)
                    else:
                        intersection_length = spatial_size / np.cos(np.pi - angle)

                    view_direction_index = (block[0, ...] * np.cos(angle) + block[1, ...] * np.sin(angle)) / (intersection_length / (divide_num + 2))
                    for i in range(divide_num):
                        view_continuous_weight[i] = 1 / ((view_direction_index - boundary_index[i]) ** 2 + 1e-3)
                    view_continuous_weight = view_continuous_weight / np.sum(view_continuous_weight, axis=0, keepdims=True)
                    multiview_continuous_weight.append(view_continuous_weight)
                multiview_continuous_weight = np.stack(multiview_continuous_weight, axis=0).astype(np.float32)
                block_list.append(multiview_continuous_weight)
    return block_list

if __name__ == '__main__':
    os.makedirs('./field_blocks/', exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_dataset_dir', type= str, default= '/nfs/scratch/jimmy/Totalsegmentor_Pelvis_Bone_Recon_Dataset')
    parser.add_argument('--config_file', type= str, default= '../XrayTo3DPreprocess/TotalSegmentor-hips-DRR-full.yaml')
    args = parser.parse_args()
    part_dict = {
        'part': 'hip',
        'subject_list_path': '../XrayTo3DPreprocess/totalsegmentor_hip_subjects.csv',
        'sub_parts': [
            'hip_left', 'hip_right'
        ]
    }
    config = read_config_and_load_components(args.config_file)

    block_list = generate_blocks()
    blocks = np.stack(block_list, axis=0)  # K, 3, N^3
    blocks = blocks.transpose(0, 2, 1).astype(float) / (config["ROI_properties"]["size"] - 1)  # K, N^3, 3
    field_blocks_dir = os.path.join(args.saved_dataset_dir, 'field_blocks')
    mesh_dir = os.path.join(args.saved_dataset_dir, 'mesh_data')
    os.makedirs(field_blocks_dir, exist_ok=True)
    np.save(os.path.join(field_blocks_dir, 'hip_blocks.npy'), blocks)

    subject_list_path = part_dict['subject_list_path']
    subject_list = (
        pd.read_csv(subject_list_path, header=None)
        .to_numpy()
        .flatten()
    )

    print('Start to generate SDF blocks.')
    with Pool(processes=mp.cpu_count(), maxtasksperchild=1) as pool:
        for _ in tqdm(
            pool.imap_unordered(
                partial(
                    generate_filed_blocks,
                    field_type= f'{part_dict["part"]}_sdf',
                    block_list= block_list,
                    mesh_dir= mesh_dir,
                    field_blocks_dir = field_blocks_dir
                ),
                subject_list,
            ),
            total=len(subject_list)
        ):
            pass
    print('SDF blocks done.')

    print('Start to generate OF blocks.')
    for sub_part in part_dict["sub_parts"]:
        print(f'Start to generate {sub_part}_of blocks.')
        with Pool(processes=mp.cpu_count(), maxtasksperchild=1) as pool:
            for _ in tqdm(
                    pool.imap_unordered(
                        partial(
                            generate_filed_blocks,
                            field_type=f'{sub_part}_of',
                            block_list=block_list,
                            mesh_dir= mesh_dir,
                            field_blocks_dir = field_blocks_dir
                        ),
                        subject_list,
                    ),
                    total=len(subject_list)
            ):
                pass
        for subject_id in subject_list:
            generate_filed_blocks(subject_id, field_type=f'{sub_part}_of', block_list=block_list, mesh_dir= mesh_dir, field_blocks_dir = field_blocks_dir)
        print(f'{sub_part}_of done.')

    print('Start to generate inter-space fusion weight blocks.')
    inter_space_fusion_weight_blocks_dir = os.path.join(args.saved_dataset_dir, 'inter_space_fusion_weight_blocks')
    os.makedirs(inter_space_fusion_weight_blocks_dir, exist_ok=True)
    inter_space_fusion_weight_block_list = generate_inter_space_fusion_weights_blocks()
    for k, block in enumerate(inter_space_fusion_weight_block_list):
        np.save(os.path.join(inter_space_fusion_weight_blocks_dir, f'inter_space_fusion_weight_block_{k}.npy'), block)
    print('Inter-space fusion weight blocks done.')