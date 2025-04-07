import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
sys.path.append(script_dir)

import trimesh
import argparse
import numpy as np
import torch
import time
import torch.nn as nn
from dataset import Pelvis_Bone_Recon_Dataset
from utils import convert_cuda, add_ts_dis_argument
from torch.utils.data import DataLoader
from models.SdAOF import SdAOF_Net
from skimage.measure import marching_cubes
from metrics.emd import get_earth_mover_distance, get_chamfer_distance
from monai.metrics import compute_dice

def sample_points(mesh, npoint):
    points, _ = trimesh.sample.sample_surface_even(mesh, npoint)
    points = torch.from_numpy(points).float().cuda()
    points = points.unsqueeze(0)
    return points

def eval_mesh_metrics(ref_mesh, src_mesh, npoint= 4096):
    ref_points = sample_points(ref_mesh, npoint)
    src_points = sample_points(src_mesh, npoint)
    cd = get_chamfer_distance(src_points, ref_points, 1)
    cd = cd.data.cpu().numpy()
    # Sometimes emd could be unstable to produce nan results
    emd = np.nan
    while np.isnan(emd):
        emd = get_earth_mover_distance(src_points, ref_points, 1)
        emd = emd.data.cpu().numpy()

    return cd, emd

def convert_to_one_hot(tensor):
    b, _, h, w, d = tensor.shape
    one_hot = torch.zeros(b, 3, h, w, d)

    # 找到channel维度中最大值大于0.5的位置
    max_values, max_indices = torch.max(tensor, dim=1)
    mask = max_values > 0.5

    # 对于预测结果大于0.5的位置，设置对应的one-hot向量
    indices = torch.nonzero(mask, as_tuple=True)
    one_hot[indices[0], max_indices[indices] + 1, indices[1], indices[2], indices[3]] = 1

    # 对于预测结果都小于0.5的位置，设置背景类的位置为1
    indices = torch.nonzero(~mask, as_tuple=True)
    one_hot[indices[0], 0, indices[1], indices[2], indices[3]] = 1

    return one_hot

def eval_one_epoch(model, loader, npoint=100000, out_res= 112, loss= nn.MSELoss(), eval_mesh = False, save_mesh= False, name = '', sub_part_list = ['hip_left', 'hip_right'], eval_mesh_npoint= 8192):
    model.eval()
    metrics = {key: [] for key in ['loss', 'dice']}
    if eval_mesh:
        for sub_part in sub_part_list:
            metrics[f'chamfer_distance_{sub_part}'] = []
            metrics[f'earth_mover_distance_{sub_part}'] = []
    results = []

    save_mesh_dir = f'./Eval_result/{name}/saved_mesh'
    if save_mesh:
        os.makedirs(save_mesh_dir, exist_ok= True)
    with torch.no_grad():
        for item in loader:
            item = convert_cuda(item)

            subject_id = item['subject_id'][0]
            of_gt = item['occupancy_field']  # b, 2, x, y, z
            output = model(item, is_eval=True, eval_npoint=npoint)  # B, 2, N

            _, c, _ = output.shape
            if eval_mesh:
                cd = []
                emd =  []
                for i in range(c):
                    output_of = output[0, i].data.cpu().numpy()
                    output_of = output_of.reshape((out_res, out_res, out_res))  # predict of voxel

                    eval_spacing = 1.0 * 336 / out_res

                    # padding around the voxel to ensure valid mesh extraction
                    padding_size = int((360 - 336) / eval_spacing / 2)
                    padding_voxel_size = [s + 2 * padding_size for s in output_of.shape]
                    pad_output_of = np.zeros(padding_voxel_size).astype(np.float32)
                    pad_output_of[padding_size: padding_size + output_of.shape[0],
                    padding_size: padding_size + output_of.shape[1],
                    padding_size: padding_size + output_of.shape[2]] = output_of

                    verts, faces, _, _ = marching_cubes(pad_output_of, level=0.5, spacing=(eval_spacing, eval_spacing, eval_spacing))

                    #
                    float_coordinates_offset = (360 - 336) / eval_spacing / 2 - padding_size
                    verts = verts + np.array(
                        [float_coordinates_offset, float_coordinates_offset, float_coordinates_offset])
                    pred_mesh = trimesh.base.Trimesh(verts, faces)

                    # postprocessing
                    mesh_list = pred_mesh.split(only_watertight=False)
                    max_mesh = None
                    max_cnt = -1
                    for sub_mesh in mesh_list:
                        if len(sub_mesh.vertices) > max_cnt:
                            max_cnt = len(sub_mesh.vertices)
                            max_mesh = sub_mesh
                    pred_mesh = max_mesh
                    pred_mesh = trimesh.smoothing.filter_laplacian(pred_mesh)
                    gt_mesh = trimesh.load_mesh(item['mesh_gt_path'][sub_part_list[i]][0])
                    cd_value, emd_value = eval_mesh_metrics(gt_mesh, pred_mesh, eval_mesh_npoint)
                    cd.append(cd_value)
                    emd.append(emd_value)

                    if save_mesh:
                        pred_mesh.export(os.path.join(save_mesh_dir, f'{sub_part_list[i]}_{subject_id}_pred_mesh_336.ply'))

            # To calculate other metrics, binarize the output occupancy voxel
            output = output.reshape((1, c, out_res, out_res, out_res))
            loss_value = loss(output, of_gt).item()
            of_pred = convert_to_one_hot(output)
            of_gt = convert_to_one_hot(of_gt)
            dice = compute_dice(of_pred, of_gt)
            metrics['dice'].append(dice.cpu().numpy())
            metrics['loss'].append(loss_value)

            result = {
                'subject_id': subject_id,
                'loss': loss_value,
                'dice': dice.cpu().numpy(),
            }

            msg = 'Subject id: {}, loss: {:.4}, dice: {:.4}'.format(subject_id, loss_value, dice.cpu().numpy().mean())
            if eval_mesh:
                for i in range(c):
                    metrics[f'chamfer_distance_{sub_part_list[i]}'].append(cd[i])
                    result[f'chamfer_distance_{sub_part_list[i]}'] = cd[i]
                    msg += ', cd_{}: {:.4}'.format(sub_part_list[i], cd[i])
                    metrics[f'earth_mover_distance_{sub_part_list[i]}'].append(emd[i])
                    result[f'earth_mover_distance_{sub_part_list[i]}'] = emd[i]
                    msg += ', emd_{}: {:.4}'.format(sub_part_list[i], emd[i])
            results.append(result)
            print(msg)

    metrics['dice'] = np.mean(metrics['dice'])
    metrics['loss'] = np.mean(metrics['loss'])
    if eval_mesh:
        for sub_part in sub_part_list:
            metrics[f'chamfer_distance_{sub_part}'] = np.mean(metrics[f'chamfer_distance_{sub_part}'])
            metrics[f'earth_mover_distance_{sub_part}'] = np.mean(metrics[f'earth_mover_distance_{sub_part}'])

    return metrics, results

def eval_original_postprocess(args):

    print(args)
    sub_part_dict = {
        'hip': ['hip_left', 'hip_right']
    }

    test_dst = Pelvis_Bone_Recon_Dataset(
            data_root=args.data_root,
            split='test',
            dataset_config=args.dataset_config,
            out_res=args.out_res,  # low-res evaluation is faster,
        )
    test_loader = DataLoader(
        test_dst,
        batch_size=1,
        shuffle=False,
        pin_memory=False
    )
    model = SdAOF_Net(mid_dim= args.mid_dim, spatial_division_num= test_dst.spatial_division_num, out_chnls = 2, projector_layer_num= args.projector_layer_num).cuda()


    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt["net"])

    time_stamp = time.time()

    run_dir_name =  f'{args.name}_out_res_{args.out_res}'

    metrics_val, results = eval_one_epoch(
        model,
        test_loader,
        args.eval_npoint,
        args.out_res,
        name=f'{run_dir_name}/test',
        eval_mesh= True,
        save_mesh=False,
        sub_part_list=sub_part_dict[args.part],
        eval_mesh_npoint=args.eval_mesh_npoint
    )

    print_list = []
    for result in results:
        msg = 'Subject id: {}, dice: {:.4}'.format(result["subject_id"], result["dice"].mean())
        for sub_part in sub_part_dict[args.part]:
            msg += ', cd_{}: {:.4}'.format(sub_part, result[f"chamfer_distance_{sub_part}"])
            msg += ', emd_{}: {:.4}'.format(sub_part, result[f"earth_mover_distance_{sub_part}"])
        print_list.append(msg)

    msg = f"test result, time: {int(time.time() - time_stamp)} s, resolution: {args.out_res}"
    for key in metrics_val.keys():
        val = metrics_val[key]
        msg += ', {}: {:.4}'.format(key, val)
    print(msg)
    print_list.append(msg)

    os.makedirs(f'./Eval_result/{run_dir_name}/test', exist_ok= True)
    output_txt = f'./Eval_result/{run_dir_name}/test/output.txt'
    with open(output_txt, 'w') as f:
        for line in print_list:
            print(line)
            f.write(line + '\n')
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval')
    parser = add_ts_dis_argument(parser, train=False)
    args = parser.parse_args()
    eval_original_postprocess(args)

# CUDA_VISIBLE_DEVICES=0 python eval_mse_02_20.py --name train_offline_student_ema_lr_1e-3_2_view_combine_cat_run_1 --out_res 336 --divide_num 4 --num_view 2 --mid_dim 128 --combine cat --ckpt_path /home/jchenhu/code/PIFU_XRecon/scripts/logs/train_offline_student_ema_lr_1e-3_2_view_combine_cat_run_1/ep_300.pth --eval_mesh_npoint 4096
# CUDA_VISIBLE_DEVICES=3 python eval_mse_02_20.py --name wo_ema_wo_dis_train_offline_student_ema_b_1_lr_1e-3_2_view_combine_cat  --out_res 336 --divide_num 4 --num_view 2 --mid_dim 128 --combine cat --ckpt_path /home/jchenhu/code/PIFU_XRecon/scripts/logs/wo_ema_wo_dis_train_offline_student_ema_b_1_lr_1e-3_2_view_combine_cat/ep_300.pth --eval_mesh_npoint 4096

# Reproduced
# CUDA_VISIBLE_DEVICES=3 python eval.py --name SdAOF_best_new --out_res 336 --ckpt_path /home/jchenhu/code/SdAOF/scripts/logs/SdAOF/best_epo.pth --eval_mesh_npoint 4096
# CUDA_VISIBLE_DEVICES=1 python eval.py --name SdAOF_last --out_res 336 --ckpt_path /home/jchenhu/code/SdAOF/scripts/logs/SdAOF/ep_100.pth --eval_mesh_npoint 4096