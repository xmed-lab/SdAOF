import os
import sys
import time

import math

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)
sys.path.append(script_dir)

import argparse
import numpy as np
import wandb

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import Pelvis_Bone_Recon_Dataset
from models.SdAOF import SdAOF_Net
from utils import convert_cuda, add_ts_dis_argument
from eval import eval_one_epoch

def worker_init_fn(worker_id):
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)

def lr_sche(current_step: int):
    if current_step < args.warmup_steps:  # current_step / warmup_steps * base_lr
        return 0.01 + 0.99 * float(current_step / args.warmup_steps)
    else:
        if args.lr_no_decay:
            return 1.0
        else:
            # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
            return float(math.pow(0.01, (current_step - args.warmup_steps) / args.epoch))

def lr_sche_step(current_step: int):
    if current_step < args.warmup_steps:  # current_step / warmup_steps * base_lr
        return 0.01 + 0.99 * float(current_step / args.warmup_steps)
    else:
        if args.lr_no_decay:
            return 1.0
        else:
            stage = current_step // args.step_interval
            # (num_training_steps - current_step) / (num_training_steps - warmup_steps) * base_lr
            return max(float(math.pow(args.lr_step_alpha, stage)), args.min_alpha)

def sigmoid_rampup(current, rampup_length):
    '''Exponential rampup from https://arxiv.org/abs/1610.02242'''
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def load_subnet(net, subnet_prefix, pretrained_state_dict):
    for name, param in net.named_parameters():
        if name.startswith(subnet_prefix):
            if name in pretrained_state_dict:
                param_shape = param.shape
                pretrained_param_shape = pretrained_state_dict[name].shape
                if param_shape == pretrained_param_shape:
                    param.data.copy_(pretrained_state_dict[name])
                    print(f'Successful loading of weight {name}')
                else:
                    print(f"Shape mismatch for parameter {name}. Expected: {param_shape}, Got: {pretrained_param_shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser = add_ts_dis_argument(parser)
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project=args.project, entity=args.entity, name=args.name, config=args)
        wandb.define_metric("epoch")
        wandb.define_metric("distill_individual_add_feat_loss", step_metric="epoch")
        wandb.define_metric("S_mse_loss", step_metric="epoch")
        wandb.define_metric("total_loss", step_metric="epoch")
        wandb.define_metric("val_S_mse_loss", step_metric="epoch")
        wandb.define_metric("val_dice", step_metric="epoch")
        wandb.define_metric("val_chamfer_distance_hip_left", step_metric="epoch")
        wandb.define_metric("val_chamfer_distance_hip_right", step_metric="epoch")
        wandb.define_metric("val_distill_individual_add_feat_loss", step_metric="epoch")

    print(args)

    save_dir = f'./logs/{args.name}'
    os.makedirs(save_dir, exist_ok=True)

    sub_part_dict = {
        'hip': ['hip_left', 'hip_right']
    }

    # -- initialize training dataset/loader
    train_dst = Pelvis_Bone_Recon_Dataset(
        data_root=args.data_root,
        split='train',
        dataset_config=args.dataset_config,
        npoint=args.num_points,
        sdf_sample_range=args.sdf_sample_range,
        balance_ratio=args.balance_ratio
    )
    train_loader = DataLoader(
        train_dst,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn
    )

    # -- initialize evaluation dataset/loader
    eval_loader = DataLoader(
        Pelvis_Bone_Recon_Dataset(
            data_root=args.data_root,
            split='eval',
            dataset_config=args.dataset_config,
            out_res=args.out_res,  # low-res evaluation is faster,
        ),
        batch_size=1,
        shuffle=False,
        pin_memory=False,
    )

    eval_loss_loader = DataLoader(
        Pelvis_Bone_Recon_Dataset(
            data_root=args.data_root,
            split='eval',
            dataset_config=args.dataset_config,
            npoint=args.num_points,
            sdf_sample_range=args.sdf_sample_range,
            balance_ratio=args.balance_ratio,
            is_train= True
        ),
        batch_size=1,
        shuffle=False,
        pin_memory=False,
    )

    model = SdAOF_Net(mid_dim= args.mid_dim, spatial_division_num= train_dst.spatial_division_num, out_chnls = 2, projector_layer_num= args.projector_layer_num).cuda()

    # Test baseline student model
    teacher_ckpt = torch.load(args.teacher_ckpt_path)
    load_subnet(model, 'teacher_image_encoder.', teacher_ckpt["net"])

    for param in model.teacher_image_encoder.parameters():
        param.detach_()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay= args.weight_decay,
        betas=(0.99, 0.999)
    )


    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda= lr_sche_step
    )

    mse_loss_func = nn.MSELoss()
    dis_mse_loss_func = nn.MSELoss(reduction='sum')


    time_stamp = time.time()

    distill_individual_feat_alpha = args.distill_individual_feat_alpha

    if args.resume:
        ckpt = torch.load(args.ckpt_path)
        st_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        st_epoch = 0

    for epoch in np.arange(st_epoch, args.epoch + 1):
        loss_list = {
            'S_mse_loss': [],
            'val_S_mse_loss': [],
            'total_loss': []
        }

        if args.distill_individual_feat_alpha > 0:
            loss_list['distill_individual_add_feat_loss'] = []
            loss_list['val_distill_individual_add_feat_loss'] = []

        model.train()

        for step, item in enumerate(train_loader):

            item = convert_cuda(item)
            p_pred, p_gt, T_multiscale_feats, S_multiscale_feats = model(item)

            S_mse_loss = mse_loss_func(p_pred, p_gt)

            total_loss = S_mse_loss
            loss_list['S_mse_loss'].append(S_mse_loss.item())

            if distill_individual_feat_alpha > 0:
                distill_individual_add_feat_loss_mse = 0
                for lvl in range(len(T_multiscale_feats)):
                    T_divide_feat = T_multiscale_feats[lvl].detach()
                    S_divide_feat = S_multiscale_feats[lvl]

                    T_divide_feat = T_divide_feat / torch.linalg.norm(T_divide_feat, dim=(3, 4, 5), keepdim=True)
                    S_divide_feat = S_divide_feat / torch.linalg.norm(S_divide_feat, dim=(3, 4, 5), keepdim=True)
                    lvl_dis_loss = dis_mse_loss_func(T_divide_feat, S_divide_feat)
                    lvl_dis_loss = lvl_dis_loss / (T_divide_feat.shape[0] * T_divide_feat.shape[1] * T_divide_feat.shape[2])

                    distill_individual_add_feat_loss_mse += lvl_dis_loss
                distill_individual_add_feat_loss_mse = distill_individual_add_feat_loss_mse / len(T_multiscale_feats)
                total_loss += distill_individual_feat_alpha * distill_individual_add_feat_loss_mse
                loss_list['distill_individual_add_feat_loss'].append(distill_individual_add_feat_loss_mse.item())

            loss_list['total_loss'].append(total_loss.item())
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            optimizer.zero_grad()

        lr_scheduler.step()

        msg = ' epoch: {}, time: {} s'.format(epoch, int(time.time() - time_stamp))
        for key, value in loss_list.items():
            if not ('val_' in key):
                msg += ', {}: {:.4}'.format(key, np.mean(value))
                if args.wandb:
                    wandb.log({key: np.mean(value), "epoch": epoch})
        print(msg)
        time_stamp = time.time()

        with torch.no_grad():
            model.eval()

            for step, item in enumerate(eval_loss_loader):

                item = convert_cuda(item)
                p_pred, p_gt, T_multiscale_feats, S_multiscale_feats = model(item)

                S_mse_loss = mse_loss_func(p_pred, p_gt)

                total_loss = S_mse_loss
                loss_list['val_S_mse_loss'].append(S_mse_loss.item())

                if distill_individual_feat_alpha > 0:
                    distill_individual_add_feat_loss_mse = 0
                    for lvl in range(len(T_multiscale_feats)):
                        T_divide_feat = T_multiscale_feats[lvl].detach()
                        S_divide_feat = S_multiscale_feats[lvl]
                        T_divide_feat = T_divide_feat / torch.linalg.norm(T_divide_feat, dim=(3, 4, 5), keepdim=True)
                        S_divide_feat = S_divide_feat / torch.linalg.norm(S_divide_feat, dim=(3, 4, 5), keepdim=True)
                        lvl_dis_loss = dis_mse_loss_func(T_divide_feat, S_divide_feat)
                        lvl_dis_loss = lvl_dis_loss / (T_divide_feat.shape[0] * T_divide_feat.shape[1] * T_divide_feat.shape[2])
                        distill_individual_add_feat_loss_mse += lvl_dis_loss
                    distill_individual_add_feat_loss_mse = distill_individual_add_feat_loss_mse / len(T_multiscale_feats)
                    loss_list['val_distill_individual_add_feat_loss'].append(distill_individual_add_feat_loss_mse.item())

            msg = ' -- epoch: {}, time: {} s'.format(epoch, int(time.time() - time_stamp))
            for key, value in loss_list.items():
                if 'val_' in key:
                    msg += ', {}: {:.4}'.format(key, np.mean(value))
                    if args.wandb:
                        wandb.log({key: np.mean(value), "epoch": epoch})
            print(msg)
            time_stamp = time.time()
        # -- save ckpt
        if epoch % args.save_interval == 0 or (epoch > args.epoch - 100 and epoch % 25 == 0) or epoch == args.epoch:
            check_point = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(
                check_point,
                os.path.join(save_dir, f'ep_{epoch}.pth')
            )

        best_loss = np.inf
        if ((epoch % args.eval_interval == 0) or (epoch > args.epoch - 10)):
            metrics_val, _ = eval_one_epoch(
                model,
                eval_loader,
                args.eval_npoint,
                args.out_res,
                loss= mse_loss_func,
                sub_part_list= sub_part_dict[args.part],
                eval_mesh= False
            )
            msg = f' --- epoch {epoch} val: '
            for key in metrics_val.keys():
                val = metrics_val[key]
                msg += ', {}: {:.4}'.format(key, val)
                if args.wandb:
                    wandb.log({f"val_{key}": val, "epoch": epoch})
            print(msg)
            if metrics_val['loss'] < best_loss:
                best_loss = metrics_val['loss']
                check_point = {
                    "net": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch
                }
                torch.save(
                    check_point,
                    os.path.join(save_dir, f'best_epo.pth')
                )

# CUDA_VISIBLE_DEVICES=5 python train_offline_student_multiscale_dis_02_23.py --optimizer adam --batch_size 2 --lr 5e-3 --sdf_sample_range 2 --num_points 10000 --sample_strategy bof --balance_ratio 0.5 --name dis_UNet_dual_encoder_b_2_adam_lr_5e-3_2_view_combine_cat_LAD_3_layer_proj_alpha_0.2_run_2_epo_110 --out_res 128 --divide_num 4 --num_view 2 --epoch 110 --project disUNet_multiscale --feature_lvl_sim_loss mse --save_interval 50 --eval_interval 50 --mid_dim 128 --warmup_steps 25 --lr_sche_step --step_interval 50 --lr_step_alpha 0.8 --distill_individual_feat_alpha 0.2 --combine cat --teacher_ckpt_path /home/jchenhu/code/PIFU_XRecon/scripts/logs/train_offline_teacher_UNet_V2_b1_lr_1e-2_2_view_combine_cat_wo_init/ep_100.pth --projector_layer_num 3 --wandb --enable_LAD --enable_dual_encoder

# Reproduced
# CUDA_VISIBLE_DEVICES=1 python train_SdAOFNet.py --name SdAOF --out_res 128 --save_interval 50 --eval_interval 50 --distill_individual_feat_alpha 0.2 --teacher_ckpt_path /home/jchenhu/code/SdAOF/scripts/logs/Teacher_OF_Net/best_epo.pth --wandb