import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.append(project_root)


import pickle
import argparse
import numpy as np
import pandas as pd
from Data_Preprocess.XrayTo3DPreprocess.xrayto3d_preprocess_utils import read_config_and_load_components

def normalize_drr(max_value= None):
    # max value for original drrs: 0.1219
    # max value for spatial division drrs: 0.0979
    max_list = []
    for subject_id in subject_list:
        drr_dir = os.path.join(saved_dataset_dir, f'drr_data/{subject_id}/{mode}')
        with open(os.path.join(drr_dir, f'{subject_id}_hip_drr.pickle'), 'rb') as f:
            data = pickle.load(f)

            projs = data['projections']  # K, res^2\
            angles = data['angles']

            if max_value is None:
                max_list.append(projs.max())
            else:
                projs /= max_value
                projs *= 255
                projs = np.clip(projs, 0, 255)
                projs = projs.astype(int).astype(np.uint8)

        if not max_value is None:
            save_path = os.path.join(drr_dir, f'{subject_id}_hip_drr_normalized.pickle')
            with open(save_path, 'wb') as f:
                pickle.dump({
                    'projections': projs,
                    'angles': angles
            }, f, pickle.HIGHEST_PROTOCOL)

    if max_value is None:
        return np.max(max_list)
    else:
        return
if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_dataset_dir', type=str, default='/nfs/scratch/jimmy/Totalsegmentor_Pelvis_Bone_Recon_Dataset')
    parser.add_argument("--subject_list_path", type=str, default= '../XrayTo3DPreprocess/totalsegmentor_hip_subjects.csv')
    args = parser.parse_args()

    for mode in ['original', 'spatial_division']:
        saved_dataset_dir = args.saved_dataset_dir
        subject_list = (
            pd.read_csv(args.subject_list_path, header=None)
            .to_numpy()
            .flatten()
        )
        print(f'Checking max value for {mode} drr.')
        drr_max = normalize_drr(max_value=None)
        print(f'Max value is {drr_max:.4f}, use it to perform max normalization.')
        normalize_drr(drr_max)
        print(f'Normalization done for {mode} drr.')


