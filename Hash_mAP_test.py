import torch
import torch.optim as optim
import os
import time
import utils.evaluate as evaluate
import models.resnet as resnet

from tqdm import tqdm
from loguru import logger
from data.data_loader import sample_dataloader
from utils import AverageMeter
import models.SEMICON as SEMICON

def valid(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args):
    num_classes, att_size, feat_size = args.num_classes, 1, 2048
    model = SEMICON.semicon(code_length=code_length, num_classes=num_classes, att_size=att_size, feat_size=feat_size,
                            device=args.device, pretrained=True)

    model.to(args.device)
    
    model.load_state_dict(torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/model.pth', map_location=args.device), strict=False)
    model.eval()
    # 查询代码就是生成的哈希码，长度就是训练的长度
    query_code = torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/query_code.pth')
    query_code = query_code.to(args.device)
    query_dataloader.dataset.get_onehot_targets = torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/query_targets.pth')
    B = torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/database_code.pth')
    B = B.to(args.device)
    retrieval_targets = torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/database_targets.pth')
    retrieval_targets = retrieval_targets.to(args.device)

    print("query_code", query_code.shape) # [image_nums, code_length]
    print("databaase_code", B.shape)
    print("retrieval_targets", retrieval_targets.shape)
    print("database_labels", query_dataloader.dataset.get_onehot_targets().to(args.device).shape)

    
    mAP = evaluate.mean_average_precision(
        query_code.to(args.device),  # 经过模型的查询图片code
        B,      # 训练好的数据库的code
        query_dataloader.dataset.get_onehot_targets().to(args.device),  # 查询图片的分类标签
        retrieval_targets,    # 检索的标签，就是train数据库的标签
        args.device,
        args.topk,
    )
    print("Code_Length: " + str(code_length), end="; ")
    print('[mAP:{:.5f}]'.format(mAP))