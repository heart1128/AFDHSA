# SEMICON: A Learning-to-hash Solution for Large-scale Fine-grained Image Retrieval
--------------------------
The hyper-parameter for \alpha in the paper is 0.15 (It's 0.3 in the paper and we have changed it in the code. This will result in about a 1~2-point fluctuation in mAP. If the replication error is 3 points or more, there must be an issue with the code or environment, so please check the code and environment carefully.). We have also provided some training logs for the CUB, NABirds, and Food101 datasets (Logs for ECCV 2022 version please cf SEMICON_log, we have not provide paper for SEMICON++ version). 

If you find significant discrepancies in the reproduced results, you can contact us and we will do our best to address your concerns (Please provide your log files in the email. Otherwise, we cannot determine where the problem lies.).

Paper Link: https://arxiv.org/pdf/2209.13833

## Environment

Python 3.8.5  
Pytorch 1.10.0  
torchvision 0.11.1  
numpy 1.19.2
loguru 0.5.3
tqdm 4.54.1

--------------------------
## Dataset
We use the following 5 datasets: CUB200-2011, Aircraft, VegFru, Food101 and NABirds.

--------------------------
## Train

We train our model in only one 2080Ti card, for different datasets, we provide different sample training commands:  

The CUB200-2011 dataset:

     python run.py --dataset cub-2011 --root /dataset/CUB2011/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 12,24,32,48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info 'CUB-SEMICON' --momen=0.91

     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 32 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info 'CUB-SEMICON' --momen=0.91 --pretrain

     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30 --arch semicon --batch-size 16 --max-iter 40 --code-length 32 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info 'CUB-SEMICON' --momen=0.91 --pretrain


     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 12 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info CUB-SEMICON+loopSEM+GobalEncoder+ICON --momen=0.91 --pretrain

     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30  --arch semicon --batch-size 16 --max-iter 40 --code-length 12 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info CUB-SEMICON+loopSEM+GobalEncoder-ICON+SBC --momen=0.91 --pretrain

     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30  --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 12 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info CUB-SEMICON+loopSEM+GobalEncoder_train-ICON --momen=0.91 --pretrain

     python run.py --dataset cub-2011 --root ../CUB_200_2011/CUB_200_2011 --max-epoch 30  --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info CUB-SEMICON+loopSEM+GobalEncoder_train_resume-ICON --momen=0.91 --pretrain

     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 12 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info SEMICON+loop_sem+AE+SIEM --momen=0.91 --pretrain
     ###
     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 60 --code-length 24 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info cub-200 --momen=0.91 --pretrain --resume_iter 40

     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 100 --code-length 32 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info cub-200 --momen=0.91 --pretrain --resume_iter 60

     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 100 --code-length 48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info cub-200 --momen=0.91 --pretrain --resume_iter 40

     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 100 --code-length 48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info cub-200-no-encoder --momen=0.91 --pretrain --resume_iter 0

     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 60 --code-length 32 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info cub-200-no --momen=0.91 --pretrain --resume_iter 0

     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 100 --code-length 24 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info cub200_100iter --momen=0.91 --pretrain --resume_iter 54 --resume checkpoints/cub-200/24/model.pth

     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 140 --code-length 32 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info cub200_100iter --momen=0.91 --pretrain --resume_iter 100 --resume checkpoints/cub-200/32/model.pth

Ablation

     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 12 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info Ablation-cub-200-baseline --momen=0.91 --pretrain --resume_iter 0

     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 12 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info Ablation-cub-200-no-encoder --momen=0.91 --pretrain --resume_iter 0

     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 12 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info Ablation-cub-200-no-encoder-All_local --momen=0.91 --pretrain --resume_iter 0

     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011 --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 12 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info Ablation-cub-200-encoder-All_global --momen=0.91 --pretrain --resume_iter 0

     

     

The Aircraft dataset:

     python run.py --dataset aircraft --root /dataset/aircraft/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 12,24,32,48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info 'Aircraft-SEMICON' --momen=0.91

     python run.py --dataset aircraft --root ../datasets/aircraft --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 12 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info Aircraft-SEMICON+loopSEM+GobalEncoder_train_resume-ICON --momen=0.91 --pretrain

     python run.py --dataset aircraft --root ../datasets/aircraft --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 24 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info Aircraft-SEMICON+loopSEM+GobalEncoder_train_resume-ICON --momen=0.91 --pretrain

     python run.py --dataset aircraft --root ../datasets/aircraft --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 40 --code-length 32 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info Aircraft-SEMICON+loopSEM+GobalEncoder_train_resume-ICON --momen=0.91 --pretrain

     python run.py --dataset aircraft --root ../datasets/aircraft --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 100 --code-length 48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 40 --num-samples 2000 --info Aircraft-SEMICON+loopSEM+GobalEncoder_train_resume-ICON --momen=0.91 --pretrain

The VegFru dataset:

     python run.py --dataset vegfru --root /dataset/vegfru/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 12,24,32,48 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 4000 --info 'VegFru-SEMICON' --momen=0.91

     python run.py --dataset vegfru --root ../datasets/vegfru-dataset/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 12 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 4000 --info VegFru --momen=0.91

     python run.py --dataset vegfru --root ../datasets/vegfru/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 24 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 4000 --info VegFru --momen=0.91 --pretrain

     python run.py --dataset vegfru --root ../datasets/vegfru/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 32 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 4000 --info VegFru --momen=0.91 --pretrain

The Food101 dataset:

     python run.py --dataset food101 --root /dataset/food101/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 12,24,32,48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 2000 --info 'Food101-SEMICON' --momen 0.91

     python run.py --dataset food101 --root ../datasets/food-101/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 12 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 2000 --info Food101 --momen 0.91 --pretrain

     python run.py --dataset food101 --root ../datasets/food-101/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 24 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 2000 --info Food101 --momen 0.91 
     
     python run.py --dataset food101 --root ../datasets/food-101/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 32 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 2000 --info Food101 --momen 0.91 --pretrain

     python run.py --dataset food101 --root ../datasets/food-101/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 2000 --info Food101 --momen 0.91 --pretrain

     python run.py --dataset food101 --root ../datasets/food-101/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 100 --code-length 24 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 2000 --info Food101-resume --momen 0.91 --pretrain --resume_iter 50

     python run.py --dataset food101 --root ../datasets/food-101/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 100 --code-length 32 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 2000 --info Food101-resume --momen 0.91 --pretrain

     python run.py --dataset food101 --root ../datasets/food-101/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 100 --code-length 48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 2000 --info Food101-resume --momen 0.91 --pretrain

The NAbirds dataset:
     
     python run.py --dataset nabirds --root /dataset/nabirds/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 32 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 4000 --info NAbirds --momen=0.91

     python run.py --dataset nabirds --root ../datasets/nabirds/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 32 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 4000 --info NAbirds --momen=0.91 --pretrain

     python run.py --dataset nabirds --root ../datasets/nabirds/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 70 --code-length 48 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 4000 --info NAbirds --momen=0.91 --pretrain

     python run.py --dataset nabirds --root ../datasets/nabirds/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 120 --code-length 48 --lr 5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 4000 --info NAbirds_resume --momen=0.91 --pretrain --resume ./checkpoints/NAbirds/48/model.pth --resume_iter 70

--------------------------
## Test

Taking the CUB200-2011 dataset as an example, the testing command is:  

     python run.py --dataset cub-2011 --root ../datasets/CUB_200_2011  --arch test --batch-size 16 --code-length 48 --wd 1e-4 --info 'cub-200'


