# -*- coding: utf-8 -*-
from MOEMDA import MDA_MOE   
from torch import optim, nn 
from tqdm import trange  
from utils import k_matrix, get_data, data_processing  
import dgl
import networkx as nx
import copy
import numpy as np
import torch as th
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_curve
from sklearn.model_selection import KFold
import torch.nn.functional as F
import argparse
import os
import matplotlib.pyplot as plt
import torch
import time
from datetime import datetime
from collections import defaultdict 

device = th.device("cuda:0" if th.cuda.is_available() else "cpu")  

# 配置参数
def parse_args():
    parser = argparse.ArgumentParser(description='MOEMDA训练')
    parser.add_argument('--random_seed', type=int, default=1, help='随机种子')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--wd', type=float, default=1e-3, help='权重衰减')
    parser.add_argument('--neighbor', type=int, default=20, help='KNN图的邻居数')
    parser.add_argument('--negative_rate', type=int, default=1, help='负采样率')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout率')
    parser.add_argument('--data_dir', type=str, default='./data/', help='数据目录')
    parser.add_argument('--dataset', type=str, default='HMDD v3.2', help='数据集版本 (HMDD v3.0 或 HMDD v3.2)')
    parser.add_argument('--dd2', type=bool, default=True, help='是否使用第二种疾病相似性')
    parser.add_argument('--mlp', type=list, default=[32, 1], help='MLP层的神经元数')
    parser.add_argument('--in_feats', type=int, default=32, help='输入特征维度')
    parser.add_argument('--out_feats', type=int, default=64, help='输出特征维度')
    parser.add_argument('--miRNA_number', type=int, default=495, help='miRNA数量')
    parser.add_argument('--disease_number', type=int, default=383, help='疾病数量')
    parser.add_argument('--use_hard_negative', type=bool, default=False, help='是否使用硬负样本挖掘')
    parser.add_argument('--use_far_negative', type=bool, default=True, help='是否使用远负样本挖掘')
    parser.add_argument('--disable_expert', type=int, default=-1, help='要排除的专家编号 (0:GAT, 1:GCN, 2:SAGE, 3:MLP, -1:不排除)')
    return parser.parse_args()

def print_met(list):
    print('AUC ：%.4f ' % (list[0]),
          'AUPR ：%.4f ' % (list[1]),
          'Accuracy ：%.4f ' % (list[2]),
          'precision ：%.4f ' % (list[3]),
          'recall ：%.4f ' % (list[4]),
          'f1_score ：%.4f \n' % (list[5]))

def test_single_expert(model, mm_graph, dd_graph, md_graph, miRNA_th, disease_th, test_samples, expert_idx, expert_name):
    model.eval()
    with th.no_grad():
        score, _, _, _, _ = model.single_expert_forward(mm_graph, dd_graph, md_graph, miRNA_th, disease_th, test_samples, expert_idx)
        score = score.cpu().numpy()
        
        # 计算评估指标
        true_labels = test_samples[:, 2]
        auc_score = roc_auc_score(true_labels, score)
        precision, recall, _ = precision_recall_curve(true_labels, score)
        aupr = auc(recall, precision)
        
        # 计算其他指标
        pred_labels = (score >= 0.5).astype(int)
        accuracy = accuracy_score(true_labels, pred_labels)
        precision_score_val = precision_score(true_labels, pred_labels)
        recall_score_val = recall_score(true_labels, pred_labels)
        f1 = f1_score(true_labels, pred_labels)
        
        print(f"\n{expert_name} 专家性能:")
        print(f"AUC: {auc_score:.4f}")
        print(f"AUPR: {aupr:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision_score_val:.4f}")
        print(f"Recall: {recall_score_val:.4f}")
        print(f"F1-score: {f1:.4f}")
        
        return [auc_score, aupr, accuracy, precision_score_val, recall_score_val, f1]

def train(data, args):
    # 创建带时间戳的唯一结果目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('results', f'run_{timestamp}')
    models_dir = os.path.join('models', f'run_{timestamp}')
    
    # 创建目录
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 初始化损失记录
    all_train_losses = []
    all_val_losses = []
    best_auc = 0
    best_model_path = ''
    
    # 存储每一折每一轮的验证集AUC
    all_fold_epoch_aucs = defaultdict(list)
    
    # 定义需要保存模型的 epoch 列表
    save_epochs = [1, 100, 200]
    
    all_score = []   
    kf = KFold(n_splits=kfolds, shuffle=True, random_state=1)  
    train_idx, valid_idx = [], []
    for train_index, valid_index in kf.split(data['train_samples']):
        train_idx.append(train_index)
        valid_idx.append(valid_index)
    
    for i in range(kfolds):
        fold_train_losses = []
        fold_val_losses = []
        one_score = []
        model = MDA_MOE(args).to(device)  
        optimizer = optim.AdamW(model.parameters(), weight_decay=args.wd, lr=args.lr)
        cross_entropy = nn.BCELoss()  

        miRNA = data['ms']
        disease = data['ds']
        a, b = data['train_samples'][train_idx[i]], data['train_samples'][valid_idx[i]]
        print(f'################Fold {i + 1} of {kfolds}################')
        epochs = trange(args.epochs, desc='train')
        for epoch in epochs:
            model.train()
            optimizer.zero_grad()
            mm_matrix = k_matrix(data['ms'], args.neighbor)
            dd_matrix = k_matrix(data['ds'], args.neighbor)
            mm_nx = nx.convert_matrix.from_numpy_array(mm_matrix) 
            dd_nx = nx.convert_matrix.from_numpy_array(dd_matrix)

            mm_graph = dgl.from_networkx(mm_nx)
            dd_graph = dgl.from_networkx(dd_nx)  
            md_copy = copy.deepcopy(data['train_md'])
            md_copy[:, 1] = md_copy[:, 1] + args.miRNA_number  
            md_graph = dgl.graph(
                (np.concatenate((md_copy[:, 0], md_copy[:, 1])), np.concatenate((md_copy[:, 1], md_copy[:, 0]))),
                num_nodes=args.miRNA_number + args.disease_number) 
            miRNA_th = th.Tensor(miRNA)
            disease_th = th.Tensor(disease)

            train_samples_th = th.Tensor(a).float()
            train_score, _, _, _, _, _ = model(mm_graph, dd_graph, md_graph, miRNA_th, disease_th, a) 

            train_loss = cross_entropy(th.flatten(train_score), train_samples_th[:, 2].to(device))
            fold_train_losses.append(train_loss.item())

            scoree, _, _, _, _, _ = model(mm_graph, dd_graph, md_graph, miRNA_th, disease_th, b) 
            scoree = scoree.cpu()
            scoree = scoree.detach().numpy()

            sc = data['train_samples'][valid_idx[i]]  
            sc_true = sc[:, 2]  
            aucc = roc_auc_score(sc_true, scoree) 

            val_samples_th = th.Tensor(b).float()
            scoree_tensor = th.from_numpy(scoree).float()
            val_loss = cross_entropy(th.flatten(scoree_tensor), val_samples_th[:, 2].to(device))
            fold_val_losses.append(val_loss.item())

            print("AUC=", np.round(aucc, 4), "loss=", np.round(train_loss.item(), 4))
            train_loss.backward() 
            optimizer.step()

            all_fold_epoch_aucs[i].append(aucc)

            if (epoch + 1) in save_epochs:
                model_save_path = os.path.join(models_dir, f'model_epoch_{epoch+1}_fold_{i+1}.pth')
                th.save(model.state_dict(), model_save_path)
                print(f"模型在 epoch {epoch+1}, fold {i+1} 保存到 {model_save_path}")

        
        if i == 0:
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.title('Training Loss Across Folds')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

            plt.subplot(1, 2, 2)
            plt.title('Validation Loss Across Folds')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')

        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(fold_train_losses, label=f'Fold {i+1} Train Loss', color=plt.cm.Set1(i/kfolds))
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(fold_val_losses, label=f'Fold {i+1} Validation Loss', color=plt.cm.Set1(i/kfolds))
        plt.legend()

        if i == kfolds - 1:
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'loss_curves_all_folds.png'))
            plt.close()

        model.eval()   

        scoree, _, _, _, _, _ = model(mm_graph, dd_graph, md_graph, miRNA_th, disease_th, b) 
        scoree = scoree.cpu()
        scoree = scoree.detach().numpy()

        sc = data['train_samples'][valid_idx[i]]
        sc_true = sc[:, 2]

        fpr, tpr, thresholds = roc_curve(sc_true, scoree)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]   
        print("Best threshold：{:.4f}".format(optimal_threshold))  

        # 计算auc
        aucc = roc_auc_score(sc_true, scoree)
        precision, recall, thresholds = precision_recall_curve(sc_true, scoree)
        print("AUC: {:.6f}".format(aucc))

        # 保存最佳模型
        if aucc > best_auc:
            best_auc = aucc
            best_model_path = os.path.join(models_dir, f'best_model_fold_{i+1}.pth')
            torch.save(model.state_dict(), best_model_path)

        auprc = auc(recall, precision)
        print("AUPRC: {:.6f}".format(auprc))

        scoree = scoree.ravel()  

        pred_labels = (scoree >= optimal_threshold).astype(int)
        accuracy = accuracy_score(sc_true, pred_labels)
        print("Accuracy: {:.6f}".format(accuracy))
        precision = precision_score(sc_true, pred_labels)
        print("Precision: {:.6f}".format(precision))
        recall = recall_score(sc_true, pred_labels)
        print("Recall: {:.6f}".format(recall))
        f1 = f1_score(sc_true, pred_labels)
        print("F1-score: {:.6f}".format(f1))
        one_score = [aucc, auprc, accuracy, precision, recall, f1]
        all_score.append(one_score)
    
    cv_metric = np.mean(all_score, axis=0)
    print('################5Fold Result################')
    print('AUC ：%.4f  AUPR ：%.4f  Accuracy ：%.4f  f1_score ：%.4f  precision ：%.4f  recall ：%.4f' % (
        cv_metric[0], cv_metric[1], cv_metric[2], cv_metric[5], cv_metric[4], cv_metric[3]))
    
    np.save(os.path.join(results_dir, 'all_fold_metrics.npy'), np.array(all_score))
    print(f"所有折的评估指标已保存到 {os.path.join(results_dir, 'all_fold_metrics.npy')}")

    np.save(os.path.join(results_dir, 'all_fold_epoch_aucs.npy'), dict(all_fold_epoch_aucs))
    print(f"每一折每一轮的验证集AUC已保存到 {os.path.join(results_dir, 'all_fold_epoch_aucs.npy')}")

    return scoree

if __name__ == '__main__':
    args = parse_args()
    print("Loading data...")
    data = get_data(args)
    print("Processing data...")
    data_processing(data, args)
    print("Training model...")
    train(data, args)


