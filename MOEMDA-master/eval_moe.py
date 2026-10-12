# -*- coding: utf-8 -*-

import random
import time
import argparse
import numpy as np
import torch as th

from utils import get_data
from train_moe import train


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
    parser = argparse.ArgumentParser(description="MOEMDA-MOE")

    parser.add_argument(
        "--random_seed",
        type=int,
        default=1
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=200
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--wd",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--neighbor",
        type=int,
        default=20
    )

    parser.add_argument(
        "--negative_rate",
        type=int,
        default=1
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/"
    )

    parser.add_argument(
        "--dd2",
        type=str2bool,
        default=True
    )

    parser.add_argument(
        "--mlp",
        type=list,
        default=[32, 1]
    )

    parser.add_argument(
        "--in_feats",
        type=int,
        default=32
    )

    parser.add_argument(
        "--out_feats",
        type=int,
        default=64
    )

    parser.add_argument(
        "--miRNA_number",
        type=int,
        default=0
    )

    parser.add_argument(
        "--disease_number",
        type=int,
        default=0
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="HMDD v3.0"
    )

    parser.add_argument(
        "--use_hard_negative",
        type=str2bool,
        default=False
    )

    parser.add_argument(
        "--use_far_negative",
        type=str2bool,
        default=False
    )

    parser.add_argument(
        "--single_expert",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--disable_expert",
        type=int,
        default=-1
    )

    parser.add_argument(
        "--kfolds",
        type=int,
        default=5
    )

    return parser.parse_args()


if __name__ == "__main__":

    start = time.time()

    args = parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    th.manual_seed(args.random_seed)

    if th.cuda.is_available():
        th.cuda.manual_seed(args.random_seed)
        th.cuda.manual_seed_all(args.random_seed)

    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

    print("开始加载数据...")

    data = get_data(args)

    args.miRNA_number = data["miRNA_number"]
    args.disease_number = data["disease_number"]

    print(
        f"miRNA数量: "
        f"{args.miRNA_number}"
    )

    print(
        f"疾病数量: "
        f"{args.disease_number}"
    )

    print(
        f"已知关联数量: "
        f"{len(data['md'])}"
    )

    expert_names = [
        "GCNExpert",
        "SAGEExpert",
        "AutoEncoderExpert",
        "MLPExpert"
    ]

    if args.single_expert == -1:
        print(
            "使用所有专家网络进行训练"
        )
    else:
        if (
            args.single_expert < 0
            or args.single_expert >= len(expert_names)
        ):
            raise ValueError(
                "single_expert 必须为 -1、0、1、2 或 3"
            )

        print(
            "使用单个专家网络进行训练: "
            f"{expert_names[args.single_expert]}"
        )

    train(
        data,
        args
    )

    end = time.time()

    print(
        f"总耗时: "
        f"{end - start:.2f}秒"
    )
