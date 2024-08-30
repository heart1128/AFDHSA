import torch
import torch.optim as optim
import os
import time
import utils.evaluate as evaluate
import models.resnet as resnet

import numpy as np
from tqdm import tqdm
from loguru import logger
from data.data_loader import sample_dataloader
from utils import AverageMeter
import models.SEMICON as SEMICON


def retrieval_image_index(index,
                          query_code, # [5794, 48]
                          database_code, # [5994, 48]
                          query_labels, # [5794, 200]
                          database_labels, # [5994, 200]
                          device,
                          topk=None,
                          ):

    
    # 查找第index张在数据库中的匹配情况，这个不是训练的，直接能计算
    # print(query_labels[index, :])
    retrieval = (query_labels[index, :] @ database_labels.t() > 0).float() # [5994]
    # 计算汉明距离
    hamming_dist = 0.5 * (database_code.shape[1] - query_code[index, :] @ database_code.t()) # 5994
    # print(torch.argmin(torch.argsort(hamming_dist)))
    # print(retrieval[torch.argmin(torch.argsort(hamming_dist))])
    # 找出前topk张
    # retrieval = retrieval[torch.argsort(hamming_dist)][:topk]

    # print(retrieval)
    topk_values = sorted(hamming_dist.tolist())[:10]
    # print(topk_values)
    
    topk_indices = [index for index, value in enumerate(hamming_dist.tolist()) if value in topk_values]
    # print(topk_indices)
    # for i in topk_indices:
    #     print(retrieval[i].item(), end=" ")
    return topk_indices[:10]




def test(query_dataloader, train_dataloader, retrieval_dataloader, code_length, args, image_name):
    num_classes, att_size, feat_size = args.num_classes, 1, 2048
    model = SEMICON.semicon(code_length=code_length, num_classes=num_classes, att_size=att_size, feat_size=feat_size,
                            device=args.device, pretrained=True)

    model.to(args.device)
    
    
    model.load_state_dict(torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/model.pth', map_location=args.device), strict=False)
    model.eval()
    query_code = torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/query_code.pth')
    query_code = query_code.to(args.device)
    query_dataloader.dataset.get_onehot_targets = torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/query_targets.pth')
    B = torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/database_code.pth')
    B = B.to(args.device)
    retrieval_targets = torch.load('./checkpoints/' + args.info + '/' + str(code_length) + '/database_targets.pth')
    retrieval_targets = retrieval_targets.to(args.device)
    


    index = None
    origin_path = None
    # 在查询图像数据集中找到需要查询的图像下标
    for i, query_image_path in enumerate(query_dataloader.dataset.QUERY_DATA):
        if i == 0:
            print(query_image_path)
        basename = os.path.basename(query_image_path)
        if image_name == basename:
            index = i
            origin_path = query_image_path
            break
    print("index = ", index)

    # 指定index进行查询，找到指定长度的下标
    topk_indices = retrieval_image_index(
        index,
        query_code.to(args.device),  # 经过模型的查询图片code
        B,      # 训练好的数据库的code
        query_dataloader.dataset.get_onehot_targets().to(args.device),  # 查询图片的分类标签
        retrieval_targets,    # 检索的标签，就是train数据库的标签
        args.device,
        10,     # 查询前十张
    )

    
    retrieval_image_path = []
    for idx , i in enumerate(topk_indices):
        path = retrieval_dataloader.dataset.RETRIEVAL_DATA[i]
        retrieval_image_path.append(path)    
        
    return retrieval_image_path, origin_path # 返回查询集的下标，查询是test
    

