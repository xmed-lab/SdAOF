from typing import Union
from pathlib import Path
from omegaconf import DictConfig, ListConfig, OmegaConf

ConfigType = Union[DictConfig, ListConfig]

def load_components(config: ConfigType, basepath, special_key) -> ConfigType:
    """
    update dict if the key == special_key
    return a updated dict
    """
    if config is not None and special_key in config:
        loaded_config = OmegaConf.load(basepath / config.pop(special_key))
        updated_config = OmegaConf.merge(loaded_config, config)
        return updated_config
    else:
        return config

def read_config_and_load_components(filepath, special_key="_load"):
    """read yaml from filepath and load subcomponents"""
    config_dict = OmegaConf.load(filepath)
    assert isinstance(config_dict, DictConfig)
    for key in config_dict:
        config_dict[key] = load_components(
            config_dict[key], Path(filepath).parent, special_key
        )
    return config_dict

def convert_cuda(item):
    for key in item.keys():
        if key not in ['subject_id', 'mesh_gt_path']:
            item[key] = item[key].float().cuda()
    return item

def add_ts_dis_argument(parser, train=True):
    parser.add_argument('--data_root', type=str, default='/nfs/scratch/jimmy/Totalsegmentor_Pelvis_Bone_Recon_Dataset')
    parser.add_argument('--epoch', type=int, default=110)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--out_res', type=int, default=128)
    parser.add_argument('--eval_npoint', type=int, default=100000)
    parser.add_argument('--dataset_config', choices= ['hip_info.json'], type= str, default= 'hip_info.json')
    parser.add_argument('--out_chnls', type= int, default= 2)
    parser.add_argument('--part', type= str, choices= ['hip'], default= 'hip')
    parser.add_argument('--ckpt_path', type=str, default='')
    parser.add_argument('--mid_dim', type=int, default= 128)
    parser.add_argument('--eval_mesh_npoint', type= int, default= 4096)
    parser.add_argument('--projector_layer_num', type= int, default= 3)


    if train:
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--lr', type=float, default=5e-3)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--num_points', type=int, default=10000)
        parser.add_argument('--sdf_sample_range', type= float, default= 2)
        parser.add_argument('--balance_ratio', type= float, default= 0.5)
        parser.add_argument('--eval_interval', type= int, default= 20)
        parser.add_argument('--save_interval', type=int, default=100)
        parser.add_argument('--distill_individual_feat_alpha', type= float, default= 0.2)
        parser.add_argument('--warmup_steps', type=int, default=50)
        parser.add_argument('--step_interval', type= int, default= 25)
        parser.add_argument('--teacher_ckpt_path', type= str, default= '/home/jchenhu/code/PIFU_XRecon/scripts/logs/train_offline_teacher_lr_5e-4_step_every_50_epoch_mse/ep_400.pth')
        parser.add_argument('--student_ckpt_path', type=str, default= '')
        parser.add_argument('--lr_step_alpha', type= float, default= 0.8)
        parser.add_argument('--min_alpha', type= float, default= -1.0)
        parser.add_argument('--weight_decay', type= float, default= 0)
        parser.add_argument('--lr_no_decay', action= 'store_true')
        parser.add_argument('--resume')

    # log config (use wandb)
    parser.add_argument('--wandb', action= 'store_true', help= 'Enable wandb logger')
    parser.add_argument('--project', type=str, default='SdAOF_Reproduced')
    parser.add_argument('--entity', type=str, default='herlocked')
    parser.add_argument('--name', type=str, default='')

    return parser