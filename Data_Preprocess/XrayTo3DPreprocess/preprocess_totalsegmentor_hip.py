import os
import tigre
import trimesh
import pickle
import open3d as o3d
import SimpleITK as sitk
import matplotlib.pyplot as plt
import shutil
import argparse
import pandas as pd
from tqdm import tqdm

from multiprocessing import Pool
from pathlib import Path
from skimage import measure
from tigre.utilities.geometry import Geometry


import numpy as np
from xrayto3d_preprocess_utils import (
    extract_bbox,
    get_orientation_code_itk,
    read_config_and_load_components,
    get_segmentation_labels,
    read_image,
    reorient_to,
    write_image,
)

class ConeGeometry_special(Geometry):
    '''
    Cone beam CT geometry.
    '''

    def __init__(self, data):
        Geometry.__init__(self)

        # VARIABLE                                          DESCRIPTION                    UNITS
        # -------------------------------------------------------------------------------------
        self.DSD = data['DSD'] / 1000  # Distance Source Detector      (m)
        self.DSO = data['DSO'] / 1000  # Distance Source Origin        (m)
        # Detector parameters
        self.nDetector = np.array(data['nDetector'])  # number of pixels              (px)
        self.dDetector = np.array(data['dDetector']) / 1000  # size of each pixel            (m)
        self.sDetector = self.nDetector * self.dDetector  # total size of the detector    (m)
        # Image parameters
        self.nVoxel = np.array(data['nVoxel'][::-1])  # number of voxels              (vx)
        self.dVoxel = np.array(data['dVoxel'][::-1]) / 1000  # size of each voxel            (m)
        self.sVoxel = self.nVoxel * self.dVoxel  # total size of the image       (m)

        # Offsets
        self.offOrigin = np.array(data['offOrigin'][::-1]) / 1000  # Offset of image from origin   (m)
        self.offDetector = np.array(
            [data['offDetector'][1], data['offDetector'][0], 0]) / 1000  # Offset of Detector            (m)

        # Auxiliary
        self.accuracy = data['accuracy']  # Accuracy of FWD proj          (vx/sample)  # noqa: E501
        # Mode
        self.mode = data['mode']  # parallel, cone                ...
        self.filter = data['filter']

def visualize_projections(save_dir, projections, angles):
    n_row = 1
    projections = projections.copy()
    projections = (projections - projections.min()) / (projections.max() - projections.min())

    fig, ax = plt.subplots(n_row, len(projections), constrained_layout=True)
    for i in range(len(projections)):
        angle = int((angles[i] / np.pi) * 180)
        ax[i].imshow(projections[i] * 255, cmap='gray', vmin=0, vmax=255)
        ax[i].set_title(f'{angle}')
        ax[i].axis('off')

    plt.savefig(os.path.join(save_dir), dpi=500)
    plt.close()

def visualize_spatial_division_projections(save_dir, projections, names):
    n_row = projections.shape[0] // len(names)
    projections = projections.copy()
    projections = (projections - projections.min()) / (projections.max() - projections.min())

    fig, ax = plt.subplots(n_row, len(names), constrained_layout=True)
    if n_row > 1:
        for row_id in range(n_row):
            for view_id in range(len(names)):
                ax[row_id, view_id].imshow(projections[row_id*len(names) + view_id] * 255, cmap='gray', vmin=0, vmax=255)
                ax[row_id, view_id].set_title(f'{names[view_id]}')
                ax[row_id, view_id].axis('off')
    else:
        for i in range(len(names)):
            ax[i].imshow(projections[i] * 255, cmap='gray', vmin=0, vmax=255)
            ax[i].set_title(f'{names[i]}')
            ax[i].axis('off')

    plt.savefig(os.path.join(save_dir), dpi=500)
    plt.close()

def calc_multiview_spatial_division_mask(spatial_division_num= 4):
    spatial_size = int(config["ROI_properties"]["size"] / config["ROI_properties"]["voxel_spacing"])

    points = np.mgrid[:spatial_size, :spatial_size, :spatial_size] + 0.5
    center = np.array([spatial_size // 2, spatial_size // 2, spatial_size // 2])
    center = center[..., None, None, None]
    points = points - center

    angles = np.linspace(
        0,
        np.pi,
        config["CT_TIGRE_config"]["numTrain"] + 1
    )[:-1] + config["CT_TIGRE_config"]["startAngle"] / 180 * np.pi
    multiview_masks = []
    for angle in angles:
        view_mask = np.zeros((spatial_division_num, spatial_size, spatial_size, spatial_size))
        assert angle <= np.pi

        if angle < np.pi / 4:
            intersection_length = spatial_size / np.cos(angle)
        elif angle < np.pi / 4 * 2:
            intersection_length = spatial_size / np.cos(np.pi / 2 - angle)
        elif angle < np.pi / 4 * 3:
            intersection_length = spatial_size / np.cos(angle - np.pi / 2)
        else:
            intersection_length = spatial_size / np.cos(np.pi - angle)

        view_direction_index = np.floor((points[0, ...] * np.cos(angle) + points[1, ...] * np.sin(angle)) / (
                    intersection_length / (spatial_division_num + 2)))

        order_view_direction_index = (spatial_division_num + 2) // 2 - 1 - view_direction_index
        order_view_direction_index[np.where(order_view_direction_index < 1)] = 1
        order_view_direction_index[np.where(order_view_direction_index > spatial_division_num)] = spatial_division_num
        order_view_direction_index = order_view_direction_index - 1

        for view_id in range(spatial_division_num):
            view_mask[view_id, ...] = np.where(order_view_direction_index == view_id, 1, 0)
        multiview_masks.append(view_mask)
    multiview_masks = np.stack(multiview_masks, axis=0).astype(np.float32)

    # Since we use tigre to generate drr, and its axis order is z,y,x
    multiview_masks = np.swapaxes(multiview_masks, 2, 4)

    return multiview_masks

def generate_drr_tigre(subject_id, ct_path, seg_path, config, output_path_template):

    # define paths
    ct = read_image(ct_path)
    seg = read_image(seg_path)
    roi_properties = config["ROI_properties"]

    # ct_physical_size = (336, 336, 336)
    ct_physical_size = (config["ROI_properties"]["size"], ) * 3

    labels = get_segmentation_labels(seg)
    # some scans may not have required anatomy labels
    if 1 not in labels:
        return

    ct_roi = extract_bbox(
        ct,
        seg,
        label_id=1,
        physical_size=ct_physical_size,
        padding_value=roi_properties["ct_padding"],
        verbose= False
    )

    if get_orientation_code_itk(ct_roi) != roi_properties["axcode"]:
        ct_roi = reorient_to(ct_roi, axcodes_to=roi_properties["axcode"])

    # Save roi
    out_ct_path = generate_path(
        "ct_roi", "ct_roi", subject_id, output_path_template, config
    )
    write_image(ct_roi, out_ct_path)

    # print(f"Subject {subject_id} Image Size {ct_roi.GetSize()} Spacing {np.around(ct_roi.GetSpacing(), 3)}")

    # Note that this transformation will result in axis order change
    # The order of itkimg is XYZ, and after transformation, the order of numpy array is ZYX
    # However, we keep such order because the required order for TIGRE projection is also ZYX
    ct_roi = sitk.GetArrayFromImage(ct_roi).astype(np.float32) # (288, 550, 550) -> (z, y, x)
    # Crop to bone range to get a better drr result
    ct_roi = np.clip(ct_roi, a_min=-50, a_max=1500)

    ct_roi = (ct_roi - ct_roi.min()) / (ct_roi.max() - ct_roi.min())
    ct_tigre_config = config[f"CT_TIGRE_config"]

    full_ct_drr_data = {}
    # generate angles
    full_ct_drr_data['train'] = {
        'angles': np.linspace(
            0,
            ct_tigre_config['totalAngle'] / 180 * np.pi,
            ct_tigre_config["numTrain"] +1
        )[:-1] + ct_tigre_config["startAngle"] / 180 * np.pi
    }
    # generate projections (from total ct, do not use segmentation label)
    ct_geo = ConeGeometry_special(ct_tigre_config)
    full_ct_drr_data['train']['projections'] = tigre.Ax(
        ct_roi,
        ct_geo,
        full_ct_drr_data['train']['angles'],
    )[:, ::-1, :]

    visualize_tigre_drr_path = generate_path(f"drrs", f"visualize_xray_from_ct", subject_id, output_path_template, config)
    visualize_projections(visualize_tigre_drr_path, full_ct_drr_data['train']['projections'], full_ct_drr_data['train']['angles'])
    saved_tigre_drr_path = generate_path(f"drrs", f"saved_xray_from_ct", subject_id, output_path_template, config)
    with open(saved_tigre_drr_path, 'wb') as f:
        pickle.dump(full_ct_drr_data['train'], f, pickle.HIGHEST_PROTOCOL)

def generate_spatial_division_drr(subject_id, ct_path, seg_path, config, output_path_template, multiview_masks):

    # define paths
    ct = read_image(ct_path)
    seg = read_image(seg_path)
    roi_properties = config["ROI_properties"]

    ct_physical_size = (roi_properties["size"], ) * 3

    labels = get_segmentation_labels(seg)
    # some scans may not have required anatomy labels
    if 1 not in labels:
        return

    ct_roi = extract_bbox(
        ct,
        seg,
        label_id=1,
        physical_size=ct_physical_size,
        padding_value=roi_properties["ct_padding"],
        verbose= False
    )

    if get_orientation_code_itk(ct_roi) != roi_properties["axcode"]:
        ct_roi = reorient_to(ct_roi, axcodes_to=roi_properties["axcode"])


    # print(f"Subject {subject_id} Image Size {ct_roi.GetSize()} Spacing {np.around(ct_roi.GetSpacing(), 3)}")

    # Note that this transformation will result in axis order change
    # The order of itkimg is XYZ, and after transformation, the order of numpy array is ZYX
    # However, we keep such order because the required order for TIGRE projection is also ZYX
    ct_roi = sitk.GetArrayFromImage(ct_roi).astype(np.float32) # (224, 224, 224) -> (z, y, x)
    # Crop to bone range to get a better drr result
    ct_roi = np.clip(ct_roi, a_min=-50, a_max=1500)

    ct_roi = (ct_roi - ct_roi.min()) / (ct_roi.max() - ct_roi.min())

    # Implement divide mode first
    ct_tigre_config = config["CT_TIGRE_config"]

    divide_ct_drr_data = {}
    # generate angles
    divide_ct_drr_data['train'] = {
        'projections': [],
        'angles': np.linspace(
            0,
            np.pi,
            config["CT_TIGRE_config"]["numTrain"]+1
        )[:-1] + config["CT_TIGRE_config"]["startAngle"] / 180 * np.pi
    }

    ct_geo = ConeGeometry_special(ct_tigre_config)

    for i, angle in enumerate(divide_ct_drr_data['train']['angles']):
        view_mask = multiview_masks[i, ...]
        view_projs = []
        for mask_id in range(view_mask.shape[0]): # [6 : 10]
            divide_proj = tigre.Ax(
                ct_roi * view_mask[mask_id],
                ct_geo,
                np.array([angle]),
            )[:, ::-1, :]
            view_projs.append(divide_proj)

        view_projs = np.concatenate(view_projs, axis=0)
        divide_ct_drr_data['train']['projections'].append(view_projs)

    visualize_proj = np.concatenate(divide_ct_drr_data['train']['projections'], axis= 0)
    divide_ct_drr_data['train']['projections'] = np.stack(divide_ct_drr_data['train']['projections'], axis= 0)

    visualize_xray_from_divide_mask_ct_tigre_path = generate_path("drrs","visualize_xray_from_spatial_division_ct", subject_id, output_path_template, config)
    visualize_spatial_division_projections(visualize_xray_from_divide_mask_ct_tigre_path, visualize_proj, [f'd {i}' for i in range(roi_properties["spatial_division_num"])])

    # Save results
    saved_tigre_drr_path = generate_path("drrs", f"saved_xray_from_spatial_division_ct", subject_id, output_path_template, config)
    with open(saved_tigre_drr_path, 'wb') as f:
        pickle.dump(divide_ct_drr_data['train'], f, pickle.HIGHEST_PROTOCOL)


def generate_meshes(subject_id, combined_seg_path, separate_seg_path, separate_seg_type, config, output_path_template, save_mesh= True, save_fields= True):
    '''
    Save mesh file (.ply), signed distance field and occupancy field of target physical size (.npy)
    '''

    # define paths
    seg = read_image(combined_seg_path)
    smooth_mesh_list = []
    for sub_pelvic_bone_seg_path, sub_pelvic_bone_seg_type in zip(separate_seg_path, separate_seg_type):
        sub_pelvic_bone_seg = read_image(sub_pelvic_bone_seg_path)
        roi_properties = config["ROI_properties"]

        # Our target physical size is (336, 336, 336)
        # Since some segmentation labels do not cover the complete pelvic
        # Roi boundary can cut the mesh
        # So we pad 0 around the roi boundary (expand physical size)
        # Thus, the extracted mesh can fill these boundary holes
        # seg_physical_size = (360, 360, 360)
        seg_physical_size = (roi_properties["padded_size"], ) * 3
        # print(seg_physical_size)

        labels = get_segmentation_labels(seg)
        # some scans may not have required anatomy labels
        if 1 not in labels:
            return

        seg_roi = extract_bbox(
            sub_pelvic_bone_seg,
            seg,
            label_id=1,
            physical_size=seg_physical_size,
            padding_value=roi_properties["seg_padding"],
            verbose= False
        )
        if get_orientation_code_itk(seg_roi) != roi_properties["axcode"]:
            seg_roi = reorient_to(seg_roi, axcodes_to=roi_properties["axcode"])

        # print(f"Subject {subject_id} Image Size {seg_roi.GetSize()} Spacing {np.around(seg_roi.GetSpacing(), 3)}")
        seg_roi = sitk.GetArrayFromImage(seg_roi).transpose(2,1,0).astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(seg_roi, 0.5, spacing= (roi_properties["voxel_spacing"], ) * 3)

        # We first utilize the trimesh to split mesh into subsets based on the connectivity
        mesh = trimesh.Trimesh(vertices= verts, faces= faces)
        mesh_list = mesh.split(only_watertight=False)
        max_mesh = None
        max_cnt = -1
        for sub_mesh in mesh_list:
            if len(sub_mesh.vertices) > max_cnt:
                max_cnt = len(sub_mesh.vertices)
                max_mesh = sub_mesh
        mesh = max_mesh
        mesh = trimesh.smoothing.filter_laplacian(mesh)
        smooth_mesh_list.append(mesh)

        if save_mesh:
            separate_mesh_ply_path = generate_path("mesh", f"{sub_pelvic_bone_seg_type}_mesh_ply_path", subject_id, output_path_template, config)
            mesh.export(separate_mesh_ply_path)

        if save_fields:
            # Open3D package to calculate of for separate mesh
            verts = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.faces)

            # to calculate sdf and of, we need to utilize the o3d.t.geometry, so we transform the data first
            dtype_f = o3d.core.float32
            dtype_i = o3d.core.int32
            device = o3d.core.Device('CPU:0')
            mesh = o3d.t.geometry.TriangleMesh(device= device)
            mesh.vertex.positions = o3d.core.Tensor(verts, dtype_f, device)
            mesh.triangle.indices = o3d.core.Tensor(faces, dtype_i, device)

            scene = o3d.t.geometry.RaycastingScene()
            _ = scene.add_triangles(mesh)
            grid_size = int(roi_properties['size'] / roi_properties['mesh_spacing'])
            voxels = np.mgrid[:grid_size, :grid_size, :grid_size]
            voxels = voxels.reshape(3, -1).transpose(1, 0).astype(np.float32)

            # Because we add padding, we only want volume centered in the 336 physical space
            pad_grid_half_size = int((roi_properties["padded_size"] - roi_properties["size"]) / roi_properties['mesh_spacing'] / 2)
            offset = np.array([pad_grid_half_size, pad_grid_half_size, pad_grid_half_size])
            voxels = voxels + offset[None, :]

            voxels = o3d.core.Tensor(voxels, dtype_f, device)

            occupancy_field = scene.compute_occupancy(voxels).numpy()
            occupancy_field = occupancy_field.reshape((grid_size, grid_size, grid_size))
            of_path = generate_path("mesh", f"{sub_pelvic_bone_seg_type}_of_path", subject_id, output_path_template, config)
            np.save(of_path, occupancy_field)

    combined_mesh = trimesh.util.concatenate(smooth_mesh_list)

    if save_mesh:
        combined_mesh_path = generate_path("mesh", f"hip_combined_ply_path", subject_id, output_path_template, config)
        combined_mesh.export(combined_mesh_path)

    if save_fields:
        # Open3D package to calculate sdf based on combined mesh
        verts = np.asarray(combined_mesh.vertices)
        faces = np.asarray(combined_mesh.faces)

        # to calculate sdf and of, we need to utilize the o3d.t.geometry, so we transform the data first
        dtype_f = o3d.core.float32
        dtype_i = o3d.core.int32
        device = o3d.core.Device('CPU:0')
        mesh = o3d.t.geometry.TriangleMesh(device=device)
        mesh.vertex.positions = o3d.core.Tensor(verts, dtype_f, device)
        mesh.triangle.indices = o3d.core.Tensor(faces, dtype_i, device)

        scene = o3d.t.geometry.RaycastingScene()
        _ = scene.add_triangles(mesh)
        grid_size = int(config['ROI_properties']['size'] / config['ROI_properties']['mesh_spacing'])
        voxels = np.mgrid[:grid_size, :grid_size, :grid_size]
        voxels = voxels.reshape(3, -1).transpose(1, 0).astype(np.float32)

        # Because we add padding, we only want volume centered in the 336 physical space
        pad_grid_half_size = int((config['ROI_properties']["padded_size"] - config['ROI_properties']["size"]) / config['ROI_properties']['mesh_spacing'] / 2)
        offset = np.array([pad_grid_half_size, pad_grid_half_size, pad_grid_half_size])
        voxels = voxels + offset[None, :]

        voxels = o3d.core.Tensor(voxels, dtype_f, device)

        signed_distance_field = scene.compute_signed_distance(voxels).numpy()

        signed_distance_field = signed_distance_field.reshape((grid_size, grid_size, grid_size))

        sdf_336_path = generate_path("mesh", "sdf_path", subject_id, output_path_template, config)

        np.save(sdf_336_path, signed_distance_field)

def generate_ct_seg_roi(subject_id, ct_path, seg_path, config, output_path_template):
    ct = read_image(ct_path)
    seg = read_image(seg_path)

    # print(f"Image Size {ct.GetSize()} Spacing {np.around(ct.GetSpacing(),3)}")

    # extract ROI and orient to particular orientation
    roi_properties = config["ROI_properties"]
    size = (roi_properties["size"],) * ct.GetDimension()

    labels = get_segmentation_labels(seg)
    # some scans may not have required anatomy labels
    if 1 not in labels:
        return
    ct_roi = extract_bbox(
        ct,
        seg,
        label_id=1,
        physical_size=size,
        padding_value=roi_properties["ct_padding"],
        verbose= False
    )

    if get_orientation_code_itk(ct_roi) != roi_properties["axcode"]:
        ct_roi = reorient_to(ct_roi, axcodes_to=roi_properties["axcode"])
    out_ct_path = generate_path(
        "ct_roi", "ct_roi", subject_id, output_path_template, config
    )
    write_image(ct_roi, out_ct_path)

    seg_roi = extract_bbox(
        seg,
        seg,
        label_id=1,
        physical_size=size,
        padding_value=roi_properties["seg_padding"],
        verbose= False
    )
    if get_orientation_code_itk(seg_roi) != roi_properties["axcode"]:
        seg_roi = reorient_to(seg_roi, axcodes_to=roi_properties["axcode"])

    out_seg_path = generate_path(
        "seg_roi", "seg_roi", subject_id, output_path_template, config
    )
    write_image(seg_roi, out_seg_path)


def create_directories(out_path_template, config):
    for key, out_dir in config["out_directories"].items():
        if key != "derivatives":
            Path(out_path_template.format(output_type=out_dir)).mkdir(
                exist_ok=True, parents=True
            )


def generate_path(sub_dir: str, name: str, subject_id, output_path_template, config):
    output_fileformat = config["filename_convention"]["output"]
    out_dirs = config["out_directories"]
    filename = output_fileformat[name].format(id=subject_id)
    # print(filename)
    out_path = output_path_template.format(
        output_type=out_dirs[sub_dir], output_name=filename
    )
    return out_path

def process_totalsegmentor_subject_drr_helper(subject_id: str, multiview_masks):
    # print(f"{subject_id}")

    input_fileformat = config["filename_convention"]["input"]
    subject_basepath = config["subjects"]["subject_basepath"]

    ct_path = os.path.join(subject_basepath, subject_id, input_fileformat["ct"])
    seg_path = os.path.join(subject_basepath, subject_id, input_fileformat["seg"])

    OUT_DIR_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}'
    OUT_PATH_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}/{{output_name}}'

    create_directories(OUT_DIR_TEMPLATE, config)

    generate_drr_tigre(subject_id, ct_path, seg_path, config, OUT_PATH_TEMPLATE)
    generate_spatial_division_drr(subject_id, ct_path, seg_path, config, OUT_PATH_TEMPLATE, multiview_masks)

def process_totalsegmentor_subject_roi_helper(subject_id: str):
    print(f"{subject_id}")
    # define paths
    input_fileformat = config["filename_convention"]["input"]

    subject_basepath = config["subjects"]["subject_basepath"]

    ct_path = os.path.join(subject_basepath, subject_id, input_fileformat["ct"])
    seg_path = os.path.join(subject_basepath, subject_id, input_fileformat["seg"])

    OUT_DIR_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}'
    OUT_PATH_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}/{{output_name}}'

    create_directories(OUT_DIR_TEMPLATE, config)

    generate_ct_seg_roi(subject_id, ct_path, seg_path, config, OUT_PATH_TEMPLATE)

def process_totalsegmentor_subject_mesh_helper(subject_id: str):
    # print(f"{subject_id}")
    # define paths
    input_fileformat = config["filename_convention"]["input"]

    subject_basepath = config["subjects"]["subject_basepath"]

    seg_path = os.path.join(subject_basepath, subject_id, input_fileformat["seg"])
    separate_seg_type = ["hip_left", "hip_right"]
    separate_seg_path = [os.path.join(subject_basepath, subject_id, input_fileformat[f'seg_{key}']) for key in separate_seg_type]

    OUT_DIR_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}'
    OUT_PATH_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}/{{output_name}}'

    create_directories(OUT_DIR_TEMPLATE, config)
    generate_meshes(subject_id, seg_path, separate_seg_path, separate_seg_type, config, OUT_PATH_TEMPLATE)

def move_drr(subject_id, config, output_path_template):
    # Copy DRRs and spatial division DRRs
    saved_original_tigre_drr_path = generate_path("drrs", "saved_xray_from_ct", subject_id, output_path_template, config)
    visualize_original_tigre_drr_path = generate_path("drrs", "visualize_xray_from_ct", subject_id, output_path_template, config)

    target_original_drr_dir = os.path.join(target_data_folder, f'drr_data/{subject_id}/original')
    target_original_saved_tigre_drr_path = os.path.join(target_original_drr_dir, f'{subject_id}_hip_drr.pickle')
    target_original_visualize_tigre_drr_path = os.path.join(target_original_drr_dir, f'{subject_id}_hip_drr.png')

    saved_spatial_division_tigre_drr_path = generate_path("drrs", "saved_xray_from_spatial_division_ct", subject_id, output_path_template, config)
    visualize_spatial_division_tigre_drr_path = generate_path("drrs", "visualize_xray_from_spatial_division_ct", subject_id, output_path_template, config)

    target_spatial_division_drr_dir = os.path.join(target_data_folder, f'drr_data/{subject_id}/spatial_division')
    target_spatial_division_saved_tigre_drr_path = os.path.join(target_spatial_division_drr_dir, f'{subject_id}_hip_drr.pickle')
    target_spatial_division_visualize_tigre_drr_path = os.path.join(target_spatial_division_drr_dir, f'{subject_id}_hip_drr.png')

    os.makedirs(target_original_drr_dir, exist_ok= True)
    os.makedirs(target_spatial_division_drr_dir, exist_ok=True)

    shutil.copyfile(saved_original_tigre_drr_path, target_original_saved_tigre_drr_path)
    shutil.copyfile(visualize_original_tigre_drr_path, target_original_visualize_tigre_drr_path)
    shutil.copyfile(saved_spatial_division_tigre_drr_path, target_spatial_division_saved_tigre_drr_path)
    shutil.copyfile(visualize_spatial_division_tigre_drr_path, target_spatial_division_visualize_tigre_drr_path)

def move_mesh(subject_id, config, output_path_template):
    # Copy Mesh, OF and SDF files
    sdf_path = generate_path("mesh", "sdf_path", subject_id, output_path_template, config)
    hip_left_of_path = generate_path("mesh", "hip_left_of_path", subject_id, output_path_template, config)
    hip_right_of_path = generate_path("mesh", "hip_right_of_path", subject_id, output_path_template, config)
    hip_combined_ply_path = generate_path("mesh", "hip_combined_ply_path", subject_id, output_path_template, config)
    hip_left_mesh_ply_path = generate_path("mesh", "hip_left_mesh_ply_path", subject_id, output_path_template, config)
    hip_right_mesh_ply_path = generate_path("mesh", "hip_right_mesh_ply_path", subject_id, output_path_template, config)

    target_mesh_dir = os.path.join(target_data_folder, f'mesh_data/{subject_id}')
    os.makedirs(target_mesh_dir, exist_ok=True)

    target_sdf_path = os.path.join(target_mesh_dir, f'{subject_id}_hip_sdf.npy')
    target_hip_left_of_path = os.path.join(target_mesh_dir, f'{subject_id}_hip_left_of.npy')
    target_hip_right_of_path = os.path.join(target_mesh_dir, f'{subject_id}_hip_right_of.npy')
    target_hip_combined_ply_path = os.path.join(target_mesh_dir, f'{subject_id}_hip_combined_mesh.ply')

    target_hip_left_mesh_ply_path = os.path.join(target_mesh_dir, f'{subject_id}_hip_left_mesh.ply')
    target_hip_right_mesh_ply_path = os.path.join(target_mesh_dir, f'{subject_id}_hip_right_mesh.ply')

    shutil.copyfile(sdf_path, target_sdf_path)
    shutil.copyfile(hip_left_of_path, target_hip_left_of_path)
    shutil.copyfile(hip_right_of_path, target_hip_right_of_path)
    shutil.copyfile(hip_combined_ply_path, target_hip_combined_ply_path)

    shutil.copyfile(hip_left_mesh_ply_path, target_hip_left_mesh_ply_path)
    shutil.copyfile(hip_right_mesh_ply_path, target_hip_right_mesh_ply_path)

def move_config():
    config_path = './TotalSegmentor-hips-DRR-full.yaml'
    target_config_path = os.path.join(target_data_folder, 'TotalSegmentor-hips-DRR-full.yaml')
    shutil.copyfile(config_path, target_config_path)

def move_files(subject_id: str):
    print(f"{subject_id}")

    subject_basepath = config["subjects"]["subject_basepath"]

    OUT_PATH_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}/{{output_name}}'

    move_mesh(subject_id, config, OUT_PATH_TEMPLATE)
    move_drr(subject_id, config, OUT_PATH_TEMPLATE)
    move_config()


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default= './TotalSegmentor-hips-DRR-full.yaml')
    parser.add_argument("--target_data_folder", type=str, default='/nfs/scratch/jimmy/Totalsegmentor_Pelvis_Bone_Recon_Dataset')

    args = parser.parse_args()
    config = read_config_and_load_components(args.config_file)
    target_data_folder = args.target_data_folder

    # subject_list = (
    #     pd.read_csv(config["subjects"]["subject_list"], header=None)
    #     .to_numpy()
    #     .flatten()
    # )
    subject_list = ["s0004"]

    print(f"found {len(subject_list)} subjects")

    num_workers = os.cpu_count()

    def initialize_config_for_all_workers():
        global config
        config = read_config_and_load_components(args.config_file)

    with Pool(
        processes=num_workers, initializer=initialize_config_for_all_workers
    ) as p:
        results = tqdm(
            p.map(process_totalsegmentor_subject_roi_helper, sorted(subject_list)),
            total=len(subject_list),
        )

    print('ROI extraction done. Start to extract meshes and calculate their occupancy filed / signed distance field.')

    for subject_id in tqdm(subject_list):
        # We use o3d for occupancy / signed distance field calculation
        # o3d could be stuck when combined with multi-thread processing
        process_totalsegmentor_subject_mesh_helper(subject_id)

    print('Mesh extraction, of, and sdf calculation done. Start to simulate DRRs from CT image.')

    multiview_spatial_division_masks = calc_multiview_spatial_division_mask(config["ROI_properties"]["spatial_division_num"])
    for subject_id in tqdm(subject_list):
        # GPU-based DRR simulation using TIGRE
        process_totalsegmentor_subject_drr_helper(subject_id, multiview_spatial_division_masks)

    print('Preprocessing from totalsegmentor dataset done. Move necessary file to target directory for subsequent processing.')

    # Move necessary processed files: Mesh, Of field, SDF field, and DRR to the target directory for subsequent processing
    # for subject_id in subject_list:
    #     move_files(subject_id)

# CUDA_VISIBLE_DEVICES="0" python preprocess_totalsegmentor_hip.py --target_data_folder /nfs/scratch/jimmy/Totalsegmentor_Pelvis_Bone_Recon_Dataset

