import os
import sys
import time

import math

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import argparse
import numpy as np
import wandb

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import Pelvis_Bone_Recon_Dataset
from models.SdAOF import Teacher_OF_Net
from utils import convert_cuda, add_ts_dis_argument
from eval import eval_one_epoch

def worker_init_fn(worker_id):
    np.random.seed((worker_id + torch.initial_seed()) % np.iinfo(np.int32).max)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser = add_ts_dis_argument(parser)
    args = parser.parse_args()

    if args.wandb:
        wandb.init(project=args.project, entity=args.entity, name=args.name, config=args)
        wandb.define_metric("epoch")
        wandb.define_metric("T_mse_loss", step_metric="epoch")
        wandb.define_metric("total_loss", step_metric="epoch")
        wandb.define_metric("val_loss", step_metric="epoch")
        wandb.define_metric("val_dice", step_metric="epoch")

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
        dataset_config = args.dataset_config,
        npoint=args.num_points,
        sdf_sample_range= args.sdf_sample_range,
        balance_ratio= args.balance_ratio
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
    model = Teacher_OF_Net(mid_dim=args.mid_dim, spatial_division_num= train_dst.spatial_division_num, out_chnls=2).cuda()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda= lr_sche_step
    )

    mse_loss_func = nn.MSELoss()

    time_stamp = time.time()

    if args.resume:
        ckpt = torch.load(args.ckpt_path)
        st_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        st_epoch = 0

    for epoch in np.arange(st_epoch, args.epoch + 1):

        loss_list = {
            'T_mse_loss': [],
            'total_loss': []
        }

        model.train()

        for step, item in enumerate(train_loader):

            item = convert_cuda(item)
            p_pred, p_gt = model(item)

            T_mse_loss = 0
            for pred in p_pred:
                T_mse_loss += mse_loss_func(pred, p_gt)

            T_mse_loss = T_mse_loss / len(p_pred)

            total_loss = T_mse_loss
            loss_list['T_mse_loss'].append(T_mse_loss.item())

            loss_list['total_loss'].append(total_loss.item())
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        lr_scheduler.step()

        msg = 'epoch: {}, time: {} s'.format(epoch, int(time.time() - time_stamp))
        for key, value in loss_list.items():
            msg += ', {}: {:.4}'.format(key, np.mean(value))
            if args.wandb:
                wandb.log({key: np.mean(value), "epoch": epoch})
        print(msg)
        time_stamp = time.time()

        # -- save ckpt
        if epoch % args.save_interval == 0 or (epoch > args.epoch - 100 and epoch % 25 == 0):
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

        if (epoch % args.eval_interval == 0) or (epoch > args.epoch - 10):
            metrics_val, _ = eval_one_epoch(
                model,
                eval_loader,
                args.eval_npoint,
                args.out_res,
                loss= mse_loss_func,
                sub_part_list=sub_part_dict[args.part],
                eval_mesh = False
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
# Input resolution
# CUDA_VISIBLE_DEVICES=2 nohup python train_offline_teacher_02_20.py --optimizer adam --batch_size 1 --lr 1e-2 --sdf_sample_range 2 --num_points 10000 --sample_strategy bof --balance_ratio 0.5 --name train_offline_teacher_UNet_V2_b1_lr_1e-3_2_view_input_res_256 --num_view 2 --out_res 128 --combine cat --divide_num 4 --epoch 110 --project PIFU_separate_128 --save_interval 50 --eval_interval 50 --mid_dim 128 --warmup_steps 50 --lr_sche_step --step_interval 25 --lr_step_alpha 0.8 --enable_unet_teacher_v2 --input_res 256 &

# Reproduction exp
# CUDA_VISIBLE_DEVICES=2 python train_teacher_OFNet.py --name Teacher_OF_Net --out_res 128 --save_interval 50 --eval_interval 50 --wandb


# Divide num
# CUDA_VISIBLE_DEVICES=3 nohup python train_offline_teacher_02_20.py --optimizer adam --batch_size 1 --lr 1e-2 --sdf_sample_range 2 --num_points 10000 --sample_strategy bof --balance_ratio 0.5 --name train_offline_teacher_UNet_V2_b1_lr_1e-3_2_view_divide_num_6 --init_type xavier --init_gain 0.02 --num_view 2 --out_res 128 --combine cat --divide_num 6 --epoch 110 --project PIFU_separate_128 --save_interval 50 --eval_interval 50 --mid_dim 128 --warmup_steps 50 --lr_sche_step --step_interval 25 --lr_step_alpha 0.8 --enable_unet_teacher_v2 --wandb &
# CUDA_VISIBLE_DEVICES=3 nohup python train_offline_teacher_02_20.py --optimizer adam --batch_size 1 --lr 1e-2 --sdf_sample_range 2 --num_points 10000 --sample_strategy bof --balance_ratio 0.5 --name train_offline_teacher_UNet_V2_b1_lr_1e-3_2_view_divide_num_2 --num_view 2 --out_res 128 --combine cat --divide_num 2 --epoch 110 --project PIFU_separate_128 --save_interval 50 --eval_interval 50 --mid_dim 128 --warmup_steps 50 --lr_sche_step --step_interval 25 --lr_step_alpha 0.8 --enable_unet_teacher_v2 --wandb &
