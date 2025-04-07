import torch
from .auction_match import auction_match
from .pointnet2 import pointnet2_utils as pn2_utils
from .chamfer_distance import chamfer_distance

def get_earth_mover_distance(pred, gt, pcd_radius):
    idx, _ = auction_match(pred, gt)
    matched_out = pn2_utils.gather_operation(gt.transpose(1, 2).contiguous(), idx)
    matched_out = matched_out.transpose(1, 2).contiguous()
    dist2 = (pred - matched_out) ** 2
    dist2 = torch.sum(dist2, dim= -1)
    dist2 = torch.sqrt(dist2)
    dist2 /= pcd_radius
    return torch.mean(dist2)

def get_chamfer_distance(pred, gt, pcd_radius):
    cost_for, cost_bac = chamfer_distance(gt, pred)
    cost_for = torch.sqrt(cost_for)
    cost_bac = torch.sqrt(cost_bac)
    cost = 0.5 * cost_for + 0.5 * cost_bac
    cost /= pcd_radius
    cost = torch.mean(cost)
    return cost
