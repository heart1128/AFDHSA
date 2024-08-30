cd /home/heart/ljx/SEMICON
echo "into SEMICON dir"

echo "run python"
/home/heart/anaconda3/envs/pytorch1.10/bin/python run.py --dataset food101 --root ../datasets/food-101/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 32 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 2000 --info Food101 --momen 0.91 --pretrain
/home/heart/anaconda3/envs/pytorch1.10/bin/python run.py --dataset food101 --root ../datasets/food-101/ --max-epoch 30 --gpu 0 --arch semicon --batch-size 16 --max-iter 50 --code-length 48 --lr 2.5e-4 --wd 1e-4 --optim SGD --lr-step 45 --num-samples 2000 --info Food101 --momen 0.91 --pretrain