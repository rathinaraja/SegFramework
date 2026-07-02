Input_dataset
-------------
/training/patches  (440 images)
/training/masks  (440 images)
/test/patches  (86 images)
/test/masks  (86 images)

/training/patches, /training/masks, /test/patches are used during training
/test/masks are used after training completion

/input_path_split_files/ consists of the following split files
   |-split_train_val_test_SRC_fold1.csv
   |-split_train_val_test_SRC_fold2.csv
   |-split_train_val_test_SRC_fold3.csv
   |-split_train_val_test_SRC_fold4.csv
   |-split_train_val_test_SRC_fold5.csv

Setup and implementation
------------------------
Step 1: Installation Dependencies
-------
$ conda create -n nnUNet python=3.10
$ conda activate nnUNet 
$ git clone https://github.com/MIC-DKFZ/nnUNet.git
$ cd nnUNet
$ pip install -e .

Step 2: Set environment variables - nnU-Net requires three paths set before any command. Add these to your ~/.bashrc:
-------
$ export nnUNet_raw="/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_raw"
$ export nnUNet_preprocessed="/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_preprocessed"
$ export nnUNet_results="/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_results"
$ mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results
$ source ~/.bashrc
$ conda activate nnUNet

This creates
/data_64T_3/Raja/CDH1/nnunet/
   |-nnUNet_preprocessed
   |-nnUNet_raw
   |-nnUNet_results

Step 3: Prepare dataset 1 
-------
set the following in prepare_dataset.py
TRAIN_PATCHES_DIR = "/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/traininng/patches"
TRAIN_MASKS_DIR   = "/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/traininng/masks"
TEST_PATCHES_DIR  = "/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/test/patches"
OUT_ROOT = Path("/data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_raw/Dataset101_SRC")

Dataset, 101, and SRC in Dataset101_SRC in OUT_ROOT are important. carefully edit the paths accordingly in the subsequence programs and commands

$ python prepare_dataset.py  // set the path to existing masks and patches

This brings
nnunet/nnUNet_raw/Dataset_101_SRC/
├── dataset.json
├── imagesTr/ (it is from traininng/patches) 
│   ├── patch_001_0000.png
│   ├── patch_002_0000.png
│   └── ...
├── imagesTs/ (it is from test/patches) 
│   ├── patch_001_0000.png
│   ├── patch_002_0000.png
│   └── ...
└── labelsTr/ (it is from traininng/masks) 
    ├── patch_001.png
    ├── patch_002.png
    └── ...

Step 4: Prepare dataset 2 - to associate input splits prepared from the training set
-------
$ python nnunet_prepare.py \
    --patches_dir  /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset/patches \
    --masks_dir    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset/masks \
    --splits_dir   /home/rajaj/Project/Alex_project/seg_framework/splits/dataset_original/split_train_val_test \
    --dataset_id   101 \
    --dataset_name SRC

nnunet/nnUNet_raw/Dataset101_SRC/
├── dataset.json
├── imagesTr/ (it is from traininng/patches)  
├── imagesTs/ (it is from test/patches)  
├── labelsTr/ (it is from traininng/masks) 
├── dataset.json (it is from traininng/masks)  
├── splits_final.json (it is from traininng/masks) 
└── test_splits.json (it is from traininng/masks) 

run the following as it is on the command line, it changes dataset.json with number of channels
$ python fix.py (checking dimension, channels, etc. check the path in the program)

$ nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity  # it checks for dataset preparation integrity
$ cp $nnUNet_raw/Dataset101_SRC/splits_final.json $nnUNet_preprocessed/Dataset101_SRC/splits_final.json

/nnunet/nnUNet_preprocessed/Dataset101_SRC/
├── gt_segmentations (contains image files)
├── nnUNetPlans_2d (preparation files)
├── dataset.json
├── dataset_fingerprint.json
├── nnUNetPlans.json
└── splits_final.json 

Step 5: Training the model
-------
chmod +x nnunet_train.sh
./nnunet_train.sh

Or with parallel GPUs:
CUDA_VISIBLE_DEVICES=0 PYTHONWARNINGS="ignore" nnUNetv2_train 101 2d 0 --npz > fold0.log &
CUDA_VISIBLE_DEVICES=1 PYTHONWARNINGS="ignore" nnUNetv2_train 101 2d 1 --npz > fold1.log &
CUDA_VISIBLE_DEVICES=2 PYTHONWARNINGS="ignore" nnUNetv2_train 101 2d 2 --npz > fold2.log &
CUDA_VISIBLE_DEVICES=3 PYTHONWARNINGS="ignore" nnUNetv2_train 101 2d 3 --npz > fold3.log &
CUDA_VISIBLE_DEVICES=4 PYTHONWARNINGS="ignore" nnUNetv2_train 101 2d 4 --npz > fold4.log &
wait

if there is any gpu/cuda error then
# Uninstall current
pip uninstall torch torchvision torchaudio -y

# Install matching version — driver 12060 supports up to CUDA 12.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

Output
------
/data_64T_3/Raja/CDH1/nnunet/nnUNet_results/Dataset101_SRC/nnUNetTrainer__nnUNetPlans__2d
|-fold0
   |-checkpoint_best.pth
   |-checkpoint_latest.pth
   |-progress.png
|-fold1
   |-checkpoint_best.pth
   |-checkpoint_latest.pth
   |-progress.png
|-fold2
   |-checkpoint_best.pth
   |-checkpoint_latest.pth
   |-progress.png
|-fold3
   |-checkpoint_best.pth
   |-checkpoint_latest.pth
   |-progress.png
|-fold4
   |-checkpoint_best.pth
   |-checkpoint_latest.pth
   |-progress.png

Step 5: Evaluation with test set
------- 
When a single file is given, only that one fold is processed and Results/fold1/ is created. When a directory is given, all valid CSVs are processed in sorted order as fold1, fold2, etc.

python nnunet_evaluate.py \
 --dataset_id    101 \
 --dataset_name  SRC \
 --fold fold0 \
 --masks_dir     /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/test/masks \
 --splits_dir    /home/rajaj/Project/Alex_project/seg_framework/splits/dataset_original/split_train_val_test/split_train_val_test_SRC_fold1.csv \
 --output_dir    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_validation_results \
 --device        cuda:1 \
 --checkpoint    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_results/Dataset101_SRC/nnUNetTrainer__nnUNetPlans__2d/fold_0/checkpoint_best.pth 

python nnunet_evaluate.py \
 --dataset_id    101 \
 --dataset_name  SRC \
 --fold fold1 \
 --masks_dir     /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/test/masks \
 --splits_dir    /home/rajaj/Project/Alex_project/seg_framework/splits/dataset_original/split_train_val_test/split_train_val_test_SRC_fold2.csv \
 --output_dir    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_validation_results \
 --device        cuda:1 \
 --checkpoint    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_results/Dataset101_SRC/nnUNetTrainer__nnUNetPlans__2d/fold_1/checkpoint_best.pth 

python nnunet_evaluate.py \
 --dataset_id    101 \
 --dataset_name  SRC \
 --fold fold2 \
 --masks_dir     /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/test/masks \
 --splits_dir    /home/rajaj/Project/Alex_project/seg_framework/splits/dataset_original/split_train_val_test/split_train_val_test_SRC_fold3.csv \
 --output_dir    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_validation_results \
 --device        cuda:1 \
 --checkpoint    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_results/Dataset101_SRC/nnUNetTrainer__nnUNetPlans__2d/fold_2/checkpoint_best.pth 

python nnunet_evaluate.py \
 --dataset_id    101 \
 --dataset_name  SRC \
 --fold fold3 \
 --masks_dir     /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/test/masks \
 --splits_dir    /home/rajaj/Project/Alex_project/seg_framework/splits/dataset_original/split_train_val_test/split_train_val_test_SRC_fold4.csv \
 --output_dir    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_validation_results \
 --device        cuda:1 \
 --checkpoint    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_results/Dataset101_SRC/nnUNetTrainer__nnUNetPlans__2d/fold_3/checkpoint_best.pth 

python nnunet_evaluate.py \
 --dataset_id    101 \
 --dataset_name  SRC \
 --fold fold4 \
 --masks_dir     /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/dataset_split/test/masks \
 --splits_dir    /home/rajaj/Project/Alex_project/seg_framework/splits/dataset_original/split_train_val_test/split_train_val_test_SRC_fold5.csv \
 --output_dir    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_validation_results \
 --device        cuda:1 \
 --checkpoint    /data_64T_3/Raja/CDH1/src_segmentation/dataset_original/nnunet/nnUNet_results/Dataset101_SRC/nnUNetTrainer__nnUNetPlans__2d/fold_4/checkpoint_best.pth 

summary.csv should contain the following information

fold seed train_loss train_acc	train_iou train_dice val_loss	val_acc	val_iou	val_dice test_loss test_acc test_iou test_dice
1	42	0.4046	0.9017	0.7883	0.8736	0.4605	0.8807	0.7527	0.8503	0.4179	0.9008	0.7769	0.8667
2	42	0.8333	0.7658	0.6054	0.7402	0.9958	0.7006	0.4514	0.5651	0.6863	0.832	0.663	0.7822
3	42	0.2892	0.9318	0.8469	0.9129	0.5203	0.8695	0.7291	0.835	0.4937	0.8749	0.7308	0.8341
4	42	0.4169	0.8979	0.789	0.8764	0.5315	0.8734	0.7115	0.8134	0.5282	0.8755	0.723	0.8241
5	42	0.3947	0.9035	0.795	0.879	0.4524	0.8925	0.783	0.8748	0.4968	0.8779	0.7381	0.8394
mean	-	0.4677	0.8801	0.7649	0.8564	0.5921	0.8433	0.6856	0.7877	0.5246	0.8723	0.7264	0.8293
std	-	0.1884	0.0584	0.0827	0.0599	0.2043	0.0718	0.1195	0.1131	0.0886	0.0223	0.0367	0.0275

The output folder structure
/output_path/
    |-nnUNet_preprocessed
        |-Dataset101_SRC
            |-gt_segmentations
                |-*.png
            |-nnUNetPlans_2d
                |-*.b2nd
                |-*.pkl
            |-dataset.json
            |-dataset_fingerprint.json
            |-nnUNetPlans.json
            |-splits_final.json
    |-nnUNet_raw
        |-Dataset101_SRC
            |-imagesTr
                |-*.png
            |-imagesTs
                |-*.png
            |-labelsTr
                |-*.png
            |-dataset.json
            |-splits_final.json
            |-test_splits.json
    |-nnUNet_results
        |-Dataset101_SRC
            |-nnUNetTrainer__nnUNetPlans__2d
                |-fold_0
                    |-validation
                        |-*.npz
                        |-*.pkl
                        |-*.png
                    |-checkpoint_best.pth
                    |-checkpoint_final.pth
                    |-debug.json
                    |-progress.png
                    |-training_log_2026_6_25_19_18_45.txt
                |-fold_1
                |-fold_2
                |-fold_3
                |-fold_4
                |-dataset.json
                |-dataset_fingerprint.json
                |-plans.json
    |-nnUNet_validation_results
        |-fold0
            |-checkpoints
                |-best_model.pth
                |-last_model.pth
            |-predictions
                |-dataset.json
                |-plans.json
                |-predict_from_raw_data_args.json
                |-*.png
            |-visualizations
                |-*.png
            |-metrics.csv
        |-fold1
        |-fold2
        |-fold3
        |-fold4

