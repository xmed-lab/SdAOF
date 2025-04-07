The extracted dataset consists of the following files
```plaintext
  Totalsegmentor_Pelvis_Bone_Recon_Dataset/
  ├── TotalSegmentor-hips-DRR-full.yaml
  ├── mesh_data/
  │   ├── subject_id/
  │   │   └── SDF & OF files:
  │   │   ├── subject_id_hip_sdf.npy
  │   │   ├── subject_id_hip_left_of.npy
  │   │   ├── subject_id_hip_right_of.npy
  │   │   └── Mesh files:
  │   │   ├── subject_id_hip_left_mesh.ply
  │   │   ├── subject_id_hip_right_mesh.ply
  │   │   ├── subject_id_hip_combined_mesh.ply
  ├── drr_data/
  │   ├── subject_id/
  │   │   ├── original/
  │   │   │   └──Original DRR files:
  │   │   │   ├──subject_id_hip_drr.png (visulization)
  │   │   │   ├──subject_id_hip_drr.pickle
  │   │   ├── spatial_division/
  │   │   │   └──Original DRR files:
  │   │   │   ├──subject_id_hip_drr.png (visulization)
  │   │   │   ├──subject_id_hip_drr.pickle
  ```
To make it compatible for training, additional processing steps should be conducted, including
- Generate sub-volume blocks for efficient loading
- Generate weights for fusing the cross-subspace point feature as defined in Eq. (4) in our paper
    ```python
    python generate_blocks.py --saved_dataset_dir YOUR_DIRECTORY_TO_SAVED_Totalsegmentor_Pelvis_Bone_Recon_Dataset
    ```
- Normalize DRRs to 0-1
    ```
    python normalize_drr.py --saved_dataset_dir YOUR_DIRECTORY_TO_SAVED_Totalsegmentor_Pelvis_Bone_Recon_Dataset
    ```
- Move necessary config files into target directory, inclucing: ``hip_info.json``, ``hip_tigre_config.yaml``

After processing, the comprehensive structure is 
```plaintext
  Totalsegmentor_Pelvis_Bone_Recon_Dataset/
  ├── TotalSegmentor-hips-DRR-full.yaml
  ├── hip_tigre_config.yaml
  ├── hip_info.json
  ├── mesh_data/
  │   ├── subject_id/
  │   │   └── SDF & OF files:
  │   │   ├── subject_id_hip_sdf.npy
  │   │   ├── subject_id_hip_left_of.npy
  │   │   ├── subject_id_hip_right_of.npy
  │   │   └── Mesh files:
  │   │   ├── subject_id_hip_left_mesh.ply
  │   │   ├── subject_id_hip_right_mesh.ply
  │   │   ├── subject_id_hip_combined_mesh.ply
  ├── drr_data/
  │   ├── subject_id/
  │   │   ├── original/
  │   │   │   └──Original DRR files:
  │   │   │   ├──subject_id_hip_drr.png (visulization)
  │   │   │   ├──subject_id_hip_drr.pickle
  │   │   │   ├──subject_id_hip_drr_normalized.pickle
  │   │   ├── spatial_division/
  │   │   │   └──Original DRR files:
  │   │   │   ├──subject_id_hip_drr.png (visulization)
  │   │   │   ├──subject_id_hip_drr.pickle
  │   │   │   ├──subject_id_hip_drr_normalized.pickle
  ├── field_blocks/
  │   ├── subject_id/
  │   │   ├── hip_blocks.npy
  │   │   └── SDF blocks:
  │   │   ├── hip_sdf_block_0.npy
  │   │   ├── ...
  │   │   ├── hip_sdf_block_63.npy
  │   │   └── OF blocks:
  │   │   ├── hip_left_of_block_0.npy
  │   │   ├── hip_right_of_block_0.npy
  │   │   ├── ...
  │   │   ├── hip_left_of_block_0.npy
  │   │   ├── hip_right_of_block_0.npy
  ├── inter_space_fusion_weight_blocks/
  │   ├── inter_space_fusion_weight_block_0.npy
  │   ├── ...
  │   ├── inter_space_fusion_weight_block_63.npy
  ```