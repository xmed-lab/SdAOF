Here we share the complete pipeline of processing hip segmentation and CT from Totalsegmentor dataset to our biplanar mesh reconstruction dataset.

The processing includes two phases:
- Preprocess ``Totalsegmentor-hip`` to ``Totalsegmentor_Pelvis_Bone_Recon_Dataset``. We build code upon implementations from XrayTo3DPreprocess: https://github.com/naamiinepal/XrayTo3DPreprocess, and provide a minimum reproducing code. 

  Please refer to ``/XrayTo3DPreprocess`` for details.
- Preprocess ``Totalsegmentor_Pelvis_Bone_Recon_Dataset`` to make it compatible for training. 

  Please refer to ``/Totalsegmentor_Pelvis_Bone_Recon_Dataset_Preprocess`` for details.