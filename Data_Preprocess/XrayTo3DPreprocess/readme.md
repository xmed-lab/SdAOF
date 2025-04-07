# Preprocessing workflow
We bulit upon code from XrayTo3DPreprocess: https://github.com/naamiinepal/XrayTo3DPreprocess to process the mesh of hip bones from Totalsegmentor dataset. Below we share step-by-step instructions for reproducing the data processing pipeline.

# Get Statistics of Totalsegmentor-hip and generate valid subject list

- Download dataset Zenodo link:https://doi.org/10.5281/zenodo.6802613
- Collect Statistics and obtain separate hip segmentations (hip-left, hip-right)
    ```python
    python totalsegmentor_hip_stats.py --base_path YOUR_DATA_DIRECTORY_TO_TOTALSEGMENTOR_DATASET
    ``` 
- Run the jupyter notebook ``Totalsegmentor-hip-plot.ipynb``
  ```python
  python Totalsegmentor-hip-plot.py --datadir YOUR_DATA_DIRECTORY_TO_TOTALSEGMENTOR_DATASET
  ```
  
  It notebook helps you understand statistics of Totalsegmentor-hip, and produces ``totalsegmentor_hip_subjects.csv``, which include all valid files to extract hip meshes.
  The CT Scans were choosen based on whether full bone shape were available (partial hip scans were rejected manually).[]()
# Process valid subjects
- Set the directory of the processed Totalsegmentor dataset at ``subject_basepath`` in ``TotalSegmentor-hips-DRR-full.yaml``
- Run the full preprocessing pipeline for valid subjects of totalsegmentator, including
  - Mesh ROI extraction
  - Occupancy field and signed distance field calculation
  - DRR simulation (biplanar DRRs and respective spatial division DRRs)
  ```python
  CUDA_VISIBLE_DEVICES=gpu_id python preprocess_total_segmentor_hip.py --target_data_folder YOUR_DIRECTORY_TO_SAVE_PROCESSED_FILES
  ```
    In each subject folder, this will produce an additional ``/derivatives`` folder with the following structure
  ```plaintext
  derivatives/
  ├── ct_roi/
  ├── seg_roi/
  ├── mesh/
  ├── drr/
  ```
  CT and Seg ROI are intermediate files for processing purpose. The extracted Mesh, OF field, SDF field, and DRR data for valid subjects are referred to as the ``Totalsegmentor_Pelvis_Bone_Recon_Dataset``, and these extracted files are copied to the target directory. Further preprocessing for ``Totalsegmentor_Pelvis_Bone_Recon_Dataset`` is required, please refer to ``readme.md`` in ``/Totalsegmentor_Pelvis_Bone_Recon_Dataset_Preprocess``.