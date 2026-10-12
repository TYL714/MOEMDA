# -*- coding: utf-8 -*-
from MOEMDA import MDA_MOE
from torch import optim, nn
from tqdm import trange
from utils import k_matrix, get_data, make_adj, get_gaussian

import argparse
import os
import random
from datetime import datetime

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch as th

from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import KFold


device = th.device("cuda:0" if th.cuda.is_available() else "cpu")


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ("yes", "true", "t", "1"):
        return True
    if v in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="MOEMDA training")
    parser.add_argument("--random_seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--wd", type=float, default=1e-3)
    parser.add_argument("--neighbor", type=int, default=20)
    parser.add_argument("--negative_rate", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--dataset", type=str, default="HMDD v3.2")
    parser.add_argument("--dd2", type=str2bool, default=True)
    parser.add_argument("--mlp", type=list, default=[32, 1])
    parser.add_argument("--in_feats", type=int, default=32)
    parser.add_argument("--out_feats", type=int, default=64)
    parser.add_argument("--miRNA_number", type=int, default=495)
    parser.add_argument("--disease_number", type=int, default=383)
    parser.add_argument("--use_hard_negative", type=str2bool, default=False)
    parser.add_argument("--use_far_negative", type=str2bool, default=False)
    parser.add_argument("--disable_expert", type=int, default=-1)
    parser.add_argument("--kfolds", type=int, default=5)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def build_negative_pairs(data, args):
    positives = np.asarray(data["md"], dtype=np.int64)

    mask = np.ones(
        (args.miRNA_number, args.disease_number),
        dtype=bool,
    )
    mask[positives[:, 0], positives[:, 1]] = False

    negatives = np.argwhere(mask).astype(np.int64)

    rng = np.random.default_rng(args.random_seed)
    rng.shuffle(negatives)

    if args.negative_rate != -1:
        n_negative = min(
            len(negatives),
            int(args.negative_rate * len(positives)),
        )
        negatives = negatives[:n_negative]
