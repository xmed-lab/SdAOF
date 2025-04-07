import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.point_classifier import PointClassifier
from models.unet import UNet

def index_2d(feat, uv):
    # https://zhuanlan.zhihu.com/p/137271718
    # feat: [B, C, H, W]
    # uv: [B, N, 2]
    uv = uv.unsqueeze(2) # [B, N, 1, 2]
    feat = feat.transpose(2, 3) # [W, H]
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True) # [B, C, N, 1]
    return samples[:, :, :, 0] # [B, C, N]

class Projector_head(nn.Module):
    def __init__(self, input_channel, mid_channel, output_channel, layer_num):
        super().__init__()
        self.filters = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(input_channel, mid_channel, 1),
                nn.LeakyReLU(0.2),
            )
        )
        for i in range(layer_num-2):
            self.filters.append(
                nn.Sequential(
                    nn.Conv2d(mid_channel, mid_channel, 1),
                    nn.LeakyReLU(0.2),
                )
            )
        self.filters.append(
            nn.Conv2d(mid_channel, output_channel, 1),
        )

    def forward(self, x):
        for layer in self.filters:
            x = layer(x)
        return x
    
class Teacher_OF_Net(nn.Module):
    def __init__(self, mid_dim= 128, spatial_division_num= 4, out_chnls = 2):
        super().__init__()
        self.spatial_division_num = spatial_division_num
        self.teacher_image_encoder = UNet(1, mid_dim)  # Return a list of feats

        self.point_classifier = PointClassifier(
            [mid_dim * 2, 1024, 512, 256, 128, out_chnls],
            residual=True,
        )

    def forward(self, data, is_eval=False, eval_npoint=100000):
        # projection encoding
        spatial_division_drr = data['drrs'][:, :, 1: , ...] # B, M, 1+K, C, W, H

        b, m, k, c, w, h = spatial_division_drr.shape

        projs = spatial_division_drr.reshape(b * m * k, c, w, h)

        proj_feats = self.teacher_image_encoder(projs)
        proj_feats = [proj_feats]

        multi_view_feats = []
        for i in range(len(proj_feats)):
            _, c_, w_, h_ = proj_feats[i].shape
            multi_view_feats.append(proj_feats[i].reshape(b, m, k, c_, w_, h_))
        if not is_eval:
            p_pred = self.forward_points(multi_view_feats, data)
            p_gt = data['p_gt']
            return p_pred, p_gt
        else:
            total_npoint = data['proj_points'].shape[2]
            n_batch = int(np.ceil(total_npoint / eval_npoint))

            multi_view_feats = [multi_view_feats[-1]]  # Only the last level prediction is used during inference
            pred_list = []
            for i in range(n_batch):
                left = i * eval_npoint
                right = min((i + 1) * eval_npoint, total_npoint)
                p_pred = self.forward_points(
                    multi_view_feats,
                    {
                        'proj_points': data['proj_points'][..., left:right, :],
                        'spatial_weight': data['spatial_weight'][..., left:right],
                    }
                )  # B, C, N
                # During inference, we only use the last level prediction
                pred_list.append(p_pred[-1])

            pred = torch.cat(pred_list, dim=2)
            return pred


    def forward_points(self, proj_feats, data):
        # 1. query view-specific features
        p_feat_list = []

        for proj_f in proj_feats:  # B, M, K, C, W, H
            lvl_feat = []
            for i in range(2):
                feat = proj_f[:, i, ...]  # B, K, C, W, H
                p = data['proj_points'][:, i, ...]  # B, N, 2

                b, k, c, w, h = feat.shape
                p_feats = index_2d(feat.reshape(b * k, c, w, h), p.repeat(k, 1, 1))  # B * K, C, N
                p_feats = p_feats.reshape(b, k, c, -1)  # B, K, C, N
                p_feats = data['spatial_weight'][:, i, ...].unsqueeze(2) * p_feats
                p_feats = torch.sum(p_feats, dim= 1)
                lvl_feat.append(p_feats)
            p_feat_list.append(lvl_feat)

        # 2. cross-view fusion & point-wise classification

        p_pred = []
        for lvl in range(len(p_feat_list)):
            p_feats = torch.cat(p_feat_list[lvl], dim=1)  # B, C * num_views, N
            p_pred.append(self.point_classifier(p_feats))

        return p_pred

class SdAOF_Net(nn.Module):
    def __init__(self, mid_dim= 128, spatial_division_num = 4, out_chnls = 2, projector_layer_num= 3):
        super().__init__()
        self.spatial_division_num = spatial_division_num

        self.teacher_image_encoder = UNet(1, mid_dim)  # Return a list of feats
        self.student_image_encoder = UNet(1, mid_dim)  # Return a list of feats
        self.original_drr_image_encoder = UNet(1, mid_dim)

        self.student_channel_expander = nn.ModuleList()
        self.student_projectors = nn.ModuleList()
        channels = [128, 256, 512, 1024, 512, 256, 128, 64, 128]
        for channel in channels:
            self.student_channel_expander.append(
                nn.Sequential(
                    nn.Conv2d(channel, channel * self.spatial_division_num, 3, 1, 1),
                    nn.BatchNorm2d(channel * self.spatial_division_num),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(channel * self.spatial_division_num, channel * self.spatial_division_num, 3, 1, 1)
                )
            )
            self.student_projectors.append(
                Projector_head(
                    channel, channel, channel, projector_layer_num
                )
            )

        self.point_classifier = PointClassifier(
            [mid_dim * 4, 1024, 512, 256, 128, out_chnls],
            residual=True,
        )

    def forward(self, data, is_eval=False, eval_npoint=100000):


        projs = data['drrs']  # B, M, 1+K, C, W, H

        original_drr = projs[:, :, 0, ...]
        spatial_division_drr = projs[:, :, 1:, ...]

        b, m, k, c, w, h = spatial_division_drr.shape

        original_drr = original_drr.reshape(b * m, c, w, h)
        spatial_division_drr = spatial_division_drr.reshape(b * m * k, c, w, h)

        full_proj_feats, multiscale_student_feats = self.student_image_encoder.distill_forward(original_drr)
        _, c_, w_, h_ = full_proj_feats.shape

        full_drr_feats = self.original_drr_image_encoder(original_drr)
        full_drr_feats = full_drr_feats.reshape(b, m, c_, w_, h_)

        student_divide_feats = None
        for i in range(len(multiscale_student_feats)):
            _, c_, w_, h_ = multiscale_student_feats[i].shape

            student_divide_feats = self.student_channel_expander[i](multiscale_student_feats[i])
            multiscale_student_feats[i] = self.student_projectors[i](student_divide_feats.reshape(b * m * k, c_, w_, h_))
            multiscale_student_feats[i] = multiscale_student_feats[i].reshape(b, m, k, c_, w_, h_)

            if i == len(multiscale_student_feats) - 1:
                student_divide_feats = student_divide_feats.reshape(b, m, k, c_, w_, h_)

        if not is_eval:
            _, multiscale_teacher_feats = self.teacher_image_encoder.distill_forward(spatial_division_drr)

            for i in range(len(multiscale_teacher_feats)):
                _, c_, w_, h_ = multiscale_teacher_feats[i].shape
                multiscale_teacher_feats[i] = multiscale_teacher_feats[i].reshape(b, m, k, c_, w_, h_)

            s_p_pred = self.forward_points(full_drr_feats, student_divide_feats, data)
            p_gt = data['p_gt']

            return s_p_pred, p_gt, multiscale_teacher_feats, multiscale_student_feats
        else:
            total_npoint = data['proj_points'].shape[2]
            n_batch = int(np.ceil(total_npoint / eval_npoint))

            pred_list = []
            for i in range(n_batch):
                left = i * eval_npoint
                right = min((i + 1) * eval_npoint, total_npoint)
                p_pred = self.forward_points(
                    full_drr_feats,
                    student_divide_feats,
                    {
                        'proj_points': data['proj_points'][..., left:right, :],
                        'spatial_weight': data['spatial_weight'][..., left:right],
                    }
                )  # B, C, N
                # During inference, we only use the last level prediction
                pred_list.append(p_pred)

            pred = torch.cat(pred_list, dim=2)
            return pred

    def forward_points(self, full_drr_feats, divide_feats, data):
        # 1. query view-specific features
        p_feat_list = []
        for i in range(2): # biplanar view
            feat = divide_feats[:, i, ...]  # B, K, C, W, H
            p = data['proj_points'][:, i, ...]  # B, N, 2

            b, k, c, w, h = feat.shape
            p_feats = index_2d(feat.reshape(b * k, c, w, h), p.repeat(k, 1, 1))  # B * K, C, N
            p_feats = p_feats.reshape(b, k, c, -1)  # B, K, C, N
            combined_feat = data['spatial_weight'][:, i, ...].unsqueeze(2) * p_feats
            combined_feat = torch.sum(combined_feat, dim=1)

            # full projection feats

            feat = full_drr_feats[:, i, ...]  # B, C, W, H
            p_feats = torch.cat([index_2d(feat, p), combined_feat], dim=1)

            p_feat_list.append(p_feats)

        # 2. cross-view fusion & point-wise classification
        p_feats = torch.cat(p_feat_list, dim=1)  # B, C * num_views, N

        return self.point_classifier(p_feats)