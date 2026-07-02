#!/bin/bash

echo "Launching all 8 models in parallel across GPUs 0-7..."

# python train.py --config configs/attention_unet.yaml --device cuda:0 > logs/attention_unet.log 2>&1 &
# python train.py --config configs/nnunet.yaml --device cuda:1 > logs/nnunet.log 2>&1 &
# python train.py --config configs/segformer.yaml --device cuda:2 > logs/segformer.log 2>&1 &
# python train.py --config configs/segnet.yaml --device cuda:3 > logs/segnet.log 2>&1 &
# python train.py --config configs/swinunet.yaml --device cuda:4 > logs/swinunet.log 2>&1 &
# python train.py --config configs/transunet.yaml --device cuda:5 > logs/transunet.log 2>&1 &
# python train.py --config configs/unet.yaml --device cuda:6 > logs/unet.log 2>&1 &
# python train.py --config configs/unetpp.yaml --device cuda:7 > logs/unetpp.log 2>&1 &

python train.py --config configs/sam2unet.yaml --device cuda:0 > logs/sam2unet.log 2>&1 &
python train.py --config configs/mcpmedsam.yaml --device cuda:1 > logs/mcpmedsam.log 2>&1 &
python train.py --config configs/kongnet.yaml --device cuda:2 > logs/kongnet.log 2>&1 &
python train.py --config configs/swinumamba.yaml --device cuda:3 > logs/swinumamba.log 2>&1 

wait

python train.py --config configs/sambaunet.yaml --device cuda:0 > logs/sambaunet.log 2>&1  

# The 'wait' command tells the script to pause here until ALL background jobs finish
wait

echo "All 8 training runs have completed!"