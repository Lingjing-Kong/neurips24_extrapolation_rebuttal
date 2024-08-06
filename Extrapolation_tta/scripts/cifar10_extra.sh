#!/bin/bash
dataset_root="../Datasets/cifar_dataset/CIFAR-10-C/" 
dataset_source="../Datasets/cifar_dataset/cifar-10-batches-py/" 



CORRUPTION='gaussian_noise' #choose one from 'gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur' 'glass_blur' 'motion_blur' 'zoom_blur' 'snow' 'frost' 'fog' 'brightness' 'contrast' 'elastic_transform' 'pixelate' 'jpeg_compression' 
gpu=0

seed=(0 1 2)
batch_size=128
exp_group=group_name

for ((i=0;i<${#seed[@]};++i)); do
    exp_name=corruption_${CORRUPTION}_seed_${seed[i]}
    EXP=${exp_group}_seed_${seed[i]} #"$1"
    WANDB_MODE=online CUDA_VISIBLE_DEVICES=${gpu} python run_tta.py \
    --source_data_path ${dataset_source} \
    --use_source_stats \
    --target_data_path ${dataset_root} \
    --pretrained_source_path "./Source_classifiers/cifar10/ckpt.pth" \
    --dataset_name cifar10 \
    --experiment_dir "./Experiments/Online/CIFAR10_C/${EXP}/${CORRUPTION}/" \
    --seed ${seed[i]} \
    --batch_size ${batch_size} \
    --n_epochs 1 \
    --bn_epochs 0 \
    --arch resnet50_s \
    --corruption "${CORRUPTION}" \
    --corruption_level 5 \
    --n_neigh 1 \
    --n_classes 10 \
    --lr 1e-3 \
    --ema_momentum 0.99 \
    --ema_momentum_ 0\
    --aug_mult_easy 4 \
    --nn_queue_size 0 \
    --sub_policy_dim 2 \
    --lora_rank 64 \
    --lora_lr_factor 1 \
    --sparsity_weight 1e-5 \
    --exp_group ${exp_group} \
    --exp_name ${exp_name}
done