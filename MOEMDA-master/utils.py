import dgl
import os
import math
import random
import numpy as np
import torch as th
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import KFold
import networkx as nx


def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] != 0:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return th.LongTensor(edge_index)


def make_adj(edges, size):
    edges_tensor = th.LongTensor(edges).t()
    values = th.ones(len(edges))
    adj = th.sparse_coo_tensor(edges_tensor, values, size).to_dense().long()
    return adj


def k_matrix(matrix, k=20):
    num = matrix.shape[0]
    knn_graph = np.zeros(matrix.shape)
    idx_sort = np.argsort(-(matrix - np.eye(num)), axis=1)
    for i in range(num):
        knn_graph[i, idx_sort[i, :k + 1]] = matrix[i, idx_sort[i, :k + 1]]
        knn_graph[idx_sort[i, :k + 1], i] = matrix[idx_sort[i, :k + 1], i]
    return knn_graph + np.eye(num)


def get_gaussian(adj):
    Gaussian = np.zeros((adj.shape[0], adj.shape[0]), dtype=np.float32)
    gamaa = 1
    sumnorm = 0
    for i in range(adj.shape[0]):
        norm = np.linalg.norm(adj[i]) ** 2
        sumnorm = sumnorm + norm
    gama = gamaa / (sumnorm / adj.shape[0])
    for i in range(adj.shape[0]):
        for j in range(adj.shape[0]):
            Gaussian[i, j] = math.exp(-gama * (np.linalg.norm(adj[i] - adj[j]) ** 2))

    return Gaussian


def get_data(args):
    try:
        if not os.path.exists(args.data_dir):
            raise FileNotFoundError(f"数据目录不存在: {args.data_dir}")
        
        dataset_path = args.data_dir
        
        files = os.listdir(dataset_path)
        print(f"数据集目录内容: {files}")
        
        if any(f.endswith('.csv') for f in files):
            print("检测到CSV文件，使用HMDD v3.0加载函数")
            data = load_hmdd_v3_data(args, dataset_path)
        elif any(f.endswith('.txt') for f in files):
            print("检测到TXT文件，使用HMDD v3.2加载函数")
            data = load_hmdd_v32_data(args, dataset_path)
        else:
            raise ValueError(f"无法识别的数据集格式，目录内容: {files}")
        
        args.miRNA_number = data['miRNA_number']
        args.disease_number = data['disease_number']
        
        print(f"成功加载数据集:")
        print(f"miRNA数量: {data['miRNA_number']}")
        print(f"疾病数量: {data['disease_number']}")
        print(f"已知关联数量: {len(data['md'])}")
        
        return data
        
    except Exception as e:
        print(f"加载数据集时发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def mine_hard_negatives(md_matrix, miRNA_features, disease_features, args, k=5):
    positive_pairs = []
    all_possible_negatives = []
    
    for i in range(md_matrix.shape[0]):
        for j in range(md_matrix.shape[1]):
            if md_matrix[i, j] >= 1:
                positive_pairs.append((i, j))
            else:
                all_possible_negatives.append((i, j))
    
    if th.is_tensor(miRNA_features):
        miRNA_features = miRNA_features.cpu().numpy()
    if th.is_tensor(disease_features):
        disease_features = disease_features.cpu().numpy()
    
    miRNA_sim = np.zeros((md_matrix.shape[0], md_matrix.shape[0]))
    disease_sim = np.zeros((md_matrix.shape[1], md_matrix.shape[1]))
    
    for i in range(md_matrix.shape[0]):
        for j in range(i, md_matrix.shape[0]):
            if i == j:
                miRNA_sim[i, j] = 1.0
            else:
                sim = np.dot(miRNA_features[i], miRNA_features[j]) / (
                    np.linalg.norm(miRNA_features[i]) * np.linalg.norm(miRNA_features[j])
                )
                miRNA_sim[i, j] = sim
                miRNA_sim[j, i] = sim
    
    for i in range(md_matrix.shape[1]):
        for j in range(i, md_matrix.shape[1]):
            if i == j:
                disease_sim[i, j] = 1.0
            else:
                sim = np.dot(disease_features[i], disease_features[j]) / (
                    np.linalg.norm(disease_features[i]) * np.linalg.norm(disease_features[j])
                )
                disease_sim[i, j] = sim
                disease_sim[j, i] = sim
    
    hard_negatives = []
    scores = [] 
    for m_idx, d_idx in all_possible_negatives:
        score = 0
        
        for pos_m_idx, pos_d_idx in positive_pairs:
            if pos_d_idx == d_idx:  # 同一疾病
                score += miRNA_sim[m_idx, pos_m_idx]
            
            if pos_m_idx == m_idx:  # 同一miRNA
                score += disease_sim[d_idx, pos_d_idx]
        
        scores.append((score, (m_idx, d_idx)))
    
    scores.sort(reverse=True)
    
    hard_neg_count = min(len(scores), int(args.negative_rate * len(positive_pairs)))
    
    for i in range(hard_neg_count):
        score, (m_idx, d_idx) = scores[i]
        hard_negatives.append([m_idx, d_idx])
    
    print(f"找到 {len(hard_negatives)} 个硬负样本")
    return hard_negatives


def mine_far_negatives(md_matrix, miRNA_features, disease_features, args, k=5):
    positive_pairs = []
    all_possible_negatives = []
    
    for i in range(md_matrix.shape[0]):
        for j in range(md_matrix.shape[1]):
            if md_matrix[i, j] >= 1:
                positive_pairs.append((i, j))
            else:
                all_possible_negatives.append((i, j))
    
    if th.is_tensor(miRNA_features):
        miRNA_features = miRNA_features.cpu().numpy()
    if th.is_tensor(disease_features):
        disease_features = disease_features.cpu().numpy()
    
    miRNA_sim = np.zeros((md_matrix.shape[0], md_matrix.shape[0]))
    disease_sim = np.zeros((md_matrix.shape[1], md_matrix.shape[1]))
    

    for i in range(md_matrix.shape[0]):
        for j in range(i, md_matrix.shape[0]):
            if i == j:
                miRNA_sim[i, j] = 1.0
            else:
                sim = np.dot(miRNA_features[i], miRNA_features[j]) / (
                    np.linalg.norm(miRNA_features[i]) * np.linalg.norm(miRNA_features[j])
                )
                miRNA_sim[i, j] = sim
                miRNA_sim[j, i] = sim
    
    for i in range(md_matrix.shape[1]):
        for j in range(i, md_matrix.shape[1]):
            if i == j:
                disease_sim[i, j] = 1.0
            else:
                # 计算余弦相似度
                sim = np.dot(disease_features[i], disease_features[j]) / (
                    np.linalg.norm(disease_features[i]) * np.linalg.norm(disease_features[j])
                )
                disease_sim[i, j] = sim
                disease_sim[j, i] = sim
    
    # 设置相似度阈值范围
    min_similarity = 0.2  
    max_similarity = 0.8  
    # 基于相似性挖掘负样本
    candidates = []
    scores = []  #
    
    # 计算miRNA-disease对的分数
    for m_idx, d_idx in all_possible_negatives:
        score = 0
        count = 0
        
        for pos_m_idx, pos_d_idx in positive_pairs:
            if pos_d_idx == d_idx:  # 同一疾病
                score += miRNA_sim[m_idx, pos_m_idx]
                count += 1
            
            if pos_m_idx == m_idx:  # 同一miRNA
                score += disease_sim[d_idx, pos_d_idx]
                count += 1
        
        if count > 0:
            avg_score = score / count
            if min_similarity <= avg_score <= max_similarity:
                candidates.append((avg_score, (m_idx, d_idx)))
    
    candidates.sort()
    
    target_count = min(len(candidates), int(args.negative_rate * len(positive_pairs)))
    start_idx = (len(candidates) - target_count) // 2
    selected_candidates = candidates[start_idx:start_idx + target_count]
    
    far_negatives = [pair for _, pair in selected_candidates]
    
    print(f"找到 {len(far_negatives)} 个负样本")
    print(f"相似度范围: [{min_similarity:.2f}, {max_similarity:.2f}]")
    if len(selected_candidates) > 0:
        print(f"选中样本的平均相似度: {np.mean([score for score, _ in selected_candidates]):.4f}")
    
    return far_negatives


def data_processing(data, args):
    print("\n开始数据预处理...")
    print(f"处理前矩阵维度:")
    print(f"mf shape: {data['mf'].shape}")
    print(f"mfw shape: {data['mfw'].shape}")
    print(f"dss shape: {data['dss'].shape}")
    print(f"dsw shape: {data['dsw'].shape}")
    
    md_matrix = make_adj(data['md'], (args.miRNA_number, args.disease_number))
    
    one_index = []
    zero_index = []
    for i in range(md_matrix.shape[0]):
        for j in range(md_matrix.shape[1]):
            if md_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    
    # 随机种子
    random.seed(args.random_seed)
    random.shuffle(one_index)
    
    # 计算初始的高斯相似性，用于负样本挖掘
    md_numpy = md_matrix.numpy()
    initial_gm = get_gaussian(md_numpy)
    initial_gd = get_gaussian(md_numpy.transpose())
    
    if hasattr(args, 'use_hard_negative') and args.use_hard_negative:
        zero_index = mine_hard_negatives(md_numpy, initial_gm, initial_gd, args)
        unsamples = []
    elif hasattr(args, 'use_far_negative') and args.use_far_negative:
        zero_index = mine_far_negatives(md_numpy, initial_gm, initial_gd, args)
        unsamples = []
    else:
        random.shuffle(zero_index)
        if args.negative_rate == -1:
            zero_index = zero_index
        else:
            unsamples = zero_index[int(args.negative_rate * len(one_index)):]
            zero_index = zero_index[:int(args.negative_rate * len(one_index))]

    index = np.array(one_index + zero_index, int)
    label = np.array([1] * len(one_index) + [0] * len(zero_index), dtype=int)
    samples = np.concatenate((index, np.expand_dims(label, axis=1)), axis=1)
    
    data['neg_md'] = np.array(zero_index, dtype=int)
    
    md = samples[samples[:, 2] == 1, :2]
    md_matrix = make_adj(md, (args.miRNA_number, args.disease_number))
    md_matrix = md_matrix.numpy()

    gm = get_gaussian(md_matrix)
    gd = get_gaussian(md_matrix.transpose())
    
    print(f"\n高斯相似性矩阵维度:")
    print(f"gm shape: {gm.shape}")
    print(f"gd shape: {gd.shape}")
    
    ms = data['mf'] * data['mfw'] + gm * (1 - data['mfw'])
    ds = data['dss'] * data['dsw'] + gd * (1 - data['dsw'])
    
    # 更新数据字典
    data['ms'] = ms
    data['ds'] = ds
    data['train_samples'] = samples
    data['train_md'] = md
    data['unsamples'] = np.array(unsamples) if 'unsamples' in locals() else np.array([])
    
    print("\n数据预处理完成")


def predict_case(data, args):
    """处理预测案例的数据"""
    data['m_d_matrix'] = make_adj(data['md'], (args.miRNA_number, args.disease_number))
    m_d_matrix = data['m_d_matrix']
    
    one_index = []
    zero_index = []
    for i in range(m_d_matrix.shape[0]):
        for j in range(m_d_matrix.shape[1]):
            if m_d_matrix[i][j] >= 1:
                one_index.append([i, j])
            else:
                zero_index.append([i, j])
    
    random.seed(args.random_seed)
    random.shuffle(one_index)
    one_index = np.array(one_index)
    random.shuffle(zero_index)
    zero_index = np.array(zero_index)
    
    train = np.concatenate((one_index, zero_index[:int(args.negative_rate * len(one_index))]))
    
    # 确保字典中包含需要的键
    if 'mm_f' not in data:
        data['mm_f'] = data['mf']
    if 'dd_s' not in data:
        data['dd_s'] = data['dss']
    
    mm = data['mm_f'] * np.where(data['mm_f'] == 0, 0, 1) + get_gaussian(data['m_d_matrix']) * np.where(
        data['mm_f'] == 1, 0, 1)
    dd = data['dd_s'] * np.where(data['dd_s'] == 0, 0, 1) + get_gaussian(data['m_d_matrix'].t()) * np.where(
        data['dd_s'] == 1, 0, 1)
    
    data['mm'] = {'data_matrix': mm, 'edges': get_edge_index(mm)}
    data['dd'] = {'data_matrix': dd, 'edges': get_edge_index(dd)}
    data['train'] = train


def load_hmdd_v3_data(args, dataset_path):
    """加载HMDD v3.0数据集（CSV格式）"""
    try:
        print("开始加载HMDD v3.0数据...")
        
        # 读取miRNA和disease名称
        mirna_names = pd.read_csv(os.path.join(dataset_path, 'miRNA name.csv'))
        disease_names = pd.read_csv(os.path.join(dataset_path, 'disease number.csv'))
        
        miRNA_number = len(mirna_names)
        disease_number = len(disease_names)
        
        print(f"读取到 {miRNA_number} 个miRNA和 {disease_number} 个disease")
        
        # 读取关联矩阵
        md_matrix = pd.read_csv(os.path.join(dataset_path, 'm_d.csv'))
        if isinstance(md_matrix, pd.DataFrame):
            md_matrix = md_matrix.values
            
        if len(md_matrix.shape) == 2:
            if md_matrix.shape[1] > md_matrix.shape[0]:
                md_matrix = md_matrix[:, 1:]  
        
        md_associations = []
        for i in range(miRNA_number):  
            for j in range(disease_number): 
                if i < md_matrix.shape[0] and j < md_matrix.shape[1] and md_matrix[i, j] == 1:
                    md_associations.append([i, j])
        
        md_associations = np.array(md_associations)
        print(f"找到 {len(md_associations)} 个已知关联")
        
        try:
            print("尝试读取预计算的相似性矩阵...")
            mf = pd.read_csv(os.path.join(dataset_path, 'm_fs.csv'), index_col=0).values
            dss = pd.read_csv(os.path.join(dataset_path, 'd_ss.csv'), index_col=0).values
            
            print(f"相似性矩阵形状: mf={mf.shape}, dss={dss.shape}")
            
            if mf.shape[0] != miRNA_number or mf.shape[1] != miRNA_number:
                print("miRNA相似性矩阵维度不匹配，重新计算...")
                new_md_matrix = np.zeros((miRNA_number, disease_number))
                new_md_matrix[:md_matrix.shape[0], :md_matrix.shape[1]] = md_matrix
                mf = get_gaussian(new_md_matrix)
                
            if dss.shape[0] != disease_number or dss.shape[1] != disease_number:
                print("疾病相似性矩阵维度不匹配，重新计算...")
                new_md_matrix = np.zeros((miRNA_number, disease_number))
                new_md_matrix[:md_matrix.shape[0], :md_matrix.shape[1]] = md_matrix
                dss = get_gaussian(new_md_matrix.T)
                
        except Exception as e:
            print(f"读取相似性矩阵失败，使用高斯相似性计算: {e}")
            new_md_matrix = np.zeros((miRNA_number, disease_number))
            new_md_matrix[:md_matrix.shape[0], :md_matrix.shape[1]] = md_matrix
            mf = get_gaussian(new_md_matrix)
            dss = get_gaussian(new_md_matrix.T)
        
        mfw = np.ones((miRNA_number, miRNA_number)) * 0.5
        dsw = np.ones((disease_number, disease_number)) * 0.5
        
        data = {
            'miRNA_number': miRNA_number,
            'disease_number': disease_number,
            'mf': mf,
            'dss': dss,
            'mfw': mfw,
            'dsw': dsw,
            'md': md_associations,
            'm_num': mirna_names.values[:, 0],
            'd_num': disease_names.values[:, 0]
        }

        print("\n验证矩阵维度:")
        print(f"mf shape: {mf.shape}")
        print(f"mfw shape: {mfw.shape}")
        print(f"dss shape: {dss.shape}")
        print(f"dsw shape: {dsw.shape}")
        
        print("HMDD v3.0数据加载完成")
        return data

    except Exception as e:
        print(f"加载HMDD v3.0数据时发生错误: {str(e)}")
        raise


def load_hmdd_v32_data(args, dataset_path):
    """加载HMDD v3.2数据集（TXT格式）"""
    try:
        print("开始加载HMDD v3.2数据...")
        

        mirna_list = []
        with open(os.path.join(dataset_path, 'miRNA number.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    mirna_list.append(parts[1])
        miRNA_number = len(mirna_list)
        
        disease_list = []
        with open(os.path.join(dataset_path, 'disease number.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    disease_list.append(parts[1])
        disease_number = len(disease_list)
            
        print(f"读取到 {miRNA_number} 个miRNA和 {disease_number} 个disease")
            

        try:
            mf = np.loadtxt(os.path.join(dataset_path, 'miRNA functional similarity matrix.txt'))
            dss = np.loadtxt(os.path.join(dataset_path, 'disease semantic similarity matrix 1.txt'))
            
            print(f"相似性矩阵形状: mf={mf.shape}, dss={dss.shape}")
            
            # 检查维度是否匹配
            if mf.shape[0] != miRNA_number or mf.shape[1] != miRNA_number:
                print("miRNA相似性矩阵维度不匹配，重新计算...")
                md_matrix = np.zeros((miRNA_number, disease_number))
                mf = get_gaussian(md_matrix)
                
            if dss.shape[0] != disease_number or dss.shape[1] != disease_number:
                print("疾病相似性矩阵维度不匹配，重新计算...")
                md_matrix = np.zeros((miRNA_number, disease_number))
                dss = get_gaussian(md_matrix.T)
                
        except Exception as e:
            md_matrix = np.zeros((miRNA_number, disease_number))
            mf = get_gaussian(md_matrix)
            dss = get_gaussian(md_matrix.T)
        
        try:
            mfw = np.loadtxt(os.path.join(dataset_path, 'miRNA functional similarity weight matrix.txt'))
            dsw = np.loadtxt(os.path.join(dataset_path, 'disease semantic similarity weight matrix.txt'))
            
            if mfw.shape != mf.shape:
                print("miRNA权重矩阵维度不匹配，使用默认权重...")
                mfw = np.ones_like(mf) * 0.5
                
            if dsw.shape != dss.shape:
                print("疾病权重矩阵维度不匹配，使用默认权重...")
                dsw = np.ones_like(dss) * 0.5
                
        except Exception as e:
            print(f"读取权重矩阵失败: {e}")
            mfw = np.ones_like(mf) * 0.5
            dsw = np.ones_like(dss) * 0.5
        
        md_associations = []
        with open(os.path.join(dataset_path, 'known disease-miRNA association number.txt'), 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    i, j = map(lambda x: int(x)-1, parts[:2])  # 转换为0-based索引
                    if 0 <= i < miRNA_number and 0 <= j < disease_number:
                        md_associations.append([i, j])
                
        md_associations = np.array(md_associations)
        print(f"找到 {len(md_associations)} 个已知关联")
        
        data = {
            'miRNA_number': miRNA_number,
            'disease_number': disease_number,
            'mf': mf,
            'dss': dss,
            'mfw': mfw,
            'dsw': dsw,
            'md': md_associations,
            'm_num': mirna_list,
            'd_num': disease_list
        }
        
        # 验证矩阵维度
        print("\n验证矩阵维度:")
        print(f"mf shape: {mf.shape}")
        print(f"mfw shape: {mfw.shape}")
        print(f"dss shape: {dss.shape}")
        print(f"dsw shape: {dsw.shape}")
        
        print("HMDD v3.2数据加载完成")
        return data
        
    except Exception as e:
        print(f"加载HMDD v3.2数据时发生错误: {str(e)}")
        raise
