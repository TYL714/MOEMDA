# -*- coding: utf-8 -*-
import random
import time
import argparse
import torch as th
from utils import get_data, data_processing
from train_moe import train

# -*- coding: utf-8 -*-
{{ edit_1 }}
def parse_args():
    parser = argparse.ArgumentParser(description='MOEMDA-MOE')
    parser.add_argument('--random_seed', type=int, default=1, help='随机种子')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--wd', type=float, default=1e-3, help='权重衰减')
    parser.add_argument('--neighbor', type=int, default=20, help='KNN图的邻居数')
    parser.add_argument('--negative_rate', type=int, default=1, help='负采样率')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--data_dir', type=str, default='./data/', help='数据目录')
    parser.add_argument('--dd2', type=bool, default=True, help='是否使用第二种疾病相似性')
    parser.add_argument('--mlp', type=list, default=[32, 1], help='MLP层的神经元数')
    parser.add_argument('--in_feats', type=int, default=32, help='输入特征维度')
    parser.add_argument('--out_feats', type=int, default=64, help='输出特征维度')
    parser.add_argument('--miRNA_number', type=int, default=0, help='miRNA数量')
    parser.add_argument('--disease_number', type=int, default=0, help='疾病数量')
    parser.add_argument('--dataset', type=str, default='HMDD v3.0', help='数据集版本 (HMDD v3.0 或 HMDD v3.2)')
    parser.add_argument('--use_hard_negative', type=bool, default=True, help='是否使用硬负样本挖掘')
    parser.add_argument('--single_expert', type=int, default=-1, help='使用单个专家进行预测 (0:MultiHopExpert, 1:AutoEncoderExpert, 2:SAGEExpert, 3:MLPExpert, -1:使用所有专家)')
    return parser.parse_args()

if __name__ == '__main__':
    start = time.time()
    # 在这里设置随机种子，确保结果可重复
    seed = 1
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    
    print("开始加载和处理数据...")
    
    # 解析命令行参数
    args = parse_args()
    
    # 加载数据
    data = get_data(args)
    args.miRNA_number = data['miRNA_number']
    args.disease_number = data['disease_number']
    
    # 数据预处理
    data_processing(data, args)
    
    print(f"miRNA数量: {args.miRNA_number}")
    print(f"疾病数量: {args.disease_number}")
    print(f"已知关联数量: {len([x for x in data['train_samples'][:,2] if x == 1])}")
    
    # 打印专家设置
    expert_names = ['GCNExpert', 'SAGEExpert', 'AutoEncoderExpert', 'MLPExpert']
    if args.single_expert == -1:
        print("使用所有专家网络进行训练")
    else:
        print(f"使用单个专家网络进行训练: {expert_names[args.single_expert]}")
    
    # 训练并评估模型
    train(data, args)
    
    # 打印总耗时
    end = time.time()
    print(f"总耗时: {end - start:.2f}秒")

