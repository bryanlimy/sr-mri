## Data

- Data should have the following structure
    ```
    supermri/
        - dataset/
            - fastMRI/
                - multicoil_test/
                    - file_brain_AXT2_200_2000110.h5
                    - file_brain_AXT2_200_2000124.h5
                    - file_brain_AXT2_200_2000129.h5
                    - ...
                - mutlicoil_val/
                    - file_brain_AXFLAIR_200_6002462.h5
                    - file_brain_AXFLAIR_200_6002471.h5
                    - file_brain_AXFLAIR_200_6002477.h5
                    - ...
            - HCP/
                - 103818/
                    - unprocessed/
                        - 3T/
                            - T1w_MPR1/
                                - 103818_3T_AFI.nii.gz
                                - 103818_3T_BIAS_32CH.nii.gz
                                - 103818_3T_BIAS_BC.nii.gz
                                - ...
                            - T1w_MPR2/
                                - 103818_3T_AFI.nii.gz
                                - 103818_3T_BIAS_32CH.nii.gz
                                - 103818_3T_BIAS_BC.nii.gz
                                - ...
                            - T2w_SCP1/
                                - 103818_3T_AFI.nii.gz
                                - 103818_3T_BIAS_32CH.nii.gz
                                - 103818_3T_BIAS_BC.nii.gz
                                - ...
                - 105923/
                - 111312/
                - ...
            - MDS
                - affine
                    - SR_002_NHSRI_V0_affine.mat
                    - SR_002_NHSRI_V1_affine.mat
                    - SR_005_BRIC1_V0_affine.mat
                    - ...
                - rigid
                    - SR_002_NHSRI_V0_rigid.mat
                    - SR_002_NHSRI_V1_rigid.mat
                    - SR_005_BRIC1_V0_rigid.mat
                    - ...
                - warp
                    - SR_002_NHSRI_V0.mat
                    - SR_002_NHSRI_V1.mat
                    - SR_005_BRIC1_V0.mat
                    - ...
    ```

### Convert `.mat` to `.npy`
- loading `.npy` files is much faster than `.mat` files in Python, hence we provide a script to quickly convert `.mat` files to `.npy`
- the follow command convert all `.mat` scans in `MDS/rigid` to `.npy` in `MDS/rigid/npy` 
    ```
    python mat2npy.py --input_dir MDS/rigid --output_dir MDS/rigid/npy
    ```