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

    return positives, negatives


def make_samples(pos_pairs, neg_pairs, seed):
    pos_pairs = np.asarray(pos_pairs, dtype=np.int64)
    neg_pairs = np.asarray(neg_pairs, dtype=np.int64)

    pos_samples = np.concatenate(
        (
            pos_pairs,
            np.ones((len(pos_pairs), 1), dtype=np.int64),
        ),
        axis=1,
    )

    neg_samples = np.concatenate(
        (
            neg_pairs,
            np.zeros((len(neg_pairs), 1), dtype=np.int64),
        ),
        axis=1,
    )

    samples = np.concatenate(
        (pos_samples, neg_samples),
        axis=0,
    )

    rng = np.random.default_rng(seed)
    rng.shuffle(samples)

    return samples


def build_fold_data(data, train_pos, args):
    train_md_matrix = make_adj(
        train_pos,
        (args.miRNA_number, args.disease_number),
    ).numpy()

    gm = get_gaussian(train_md_matrix)
    gd = get_gaussian(train_md_matrix.T)

    ms = (
        data["mf"] * data["mfw"]
        + gm * (1.0 - data["mfw"])
    )

    ds = (
        data["dss"] * data["dsw"]
        + gd * (1.0 - data["dsw"])
    )

    mm_matrix = k_matrix(
        ms,
        args.neighbor,
    )

    dd_matrix = k_matrix(
        ds,
        args.neighbor,
    )

    mm_nx = nx.convert_matrix.from_numpy_array(mm_matrix)
    dd_nx = nx.convert_matrix.from_numpy_array(dd_matrix)

    mm_graph = dgl.from_networkx(mm_nx)
    dd_graph = dgl.from_networkx(dd_nx)

    md_copy = np.asarray(
        train_pos,
        dtype=np.int64,
    ).copy()

    md_copy[:, 1] = (
        md_copy[:, 1]
        + args.miRNA_number
    )

    src = np.concatenate(
        (
            md_copy[:, 0],
            md_copy[:, 1],
        )
    )

    dst = np.concatenate(
        (
            md_copy[:, 1],
            md_copy[:, 0],
        )
    )

    md_graph = dgl.graph(
        (src, dst),
        num_nodes=(
            args.miRNA_number
            + args.disease_number
        ),
    )

    miRNA_th = th.tensor(
        ms,
        dtype=th.float32,
    )

    disease_th = th.tensor(
        ds,
        dtype=th.float32,
    )

    return (
        mm_graph,
        dd_graph,
        md_graph,
        miRNA_th,
        disease_th,
    )


def evaluate(
    model,
    mm_graph,
    dd_graph,
    md_graph,
    miRNA_th,
    disease_th,
    samples,
):
    model.eval()

    with th.no_grad():
        score, _, _, _, _, _ = model(
            mm_graph,
            dd_graph,
            md_graph,
            miRNA_th,
            disease_th,
            samples,
        )

    score = (
        score
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    labels = samples[:, 2].astype(int)

    auc_score = roc_auc_score(
        labels,
        score,
    )

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            labels,
            score,
        )
    )

    aupr = auc(
        recall_curve,
        precision_curve,
    )

    pred = (
        score >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        pred,
    )

    precision_value = precision_score(
        labels,
        pred,
        zero_division=0,
    )

    recall_value = recall_score(
        labels,
        pred,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        pred,
        zero_division=0,
    )

    metrics = [
        auc_score,
        aupr,
        accuracy,
        precision_value,
        recall_value,
        f1,
    ]

    return metrics, score


def test_single_expert(
    model,
    mm_graph,
    dd_graph,
    md_graph,
    miRNA_th,
    disease_th,
    test_samples,
    expert_idx,
    expert_name,
):
    model.eval()

    with th.no_grad():
        score, _, _, _, _ = (
            model.single_expert_forward(
                mm_graph,
                dd_graph,
                md_graph,
                miRNA_th,
                disease_th,
                test_samples,
                expert_idx,
            )
        )

    score = (
        score
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    labels = (
        test_samples[:, 2]
        .astype(int)
    )

    auc_score = roc_auc_score(
        labels,
        score,
    )

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            labels,
            score,
        )
    )

    aupr = auc(
        recall_curve,
        precision_curve,
    )

    pred = (
        score >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        pred,
    )

    precision_value = precision_score(
        labels,
        pred,
        zero_division=0,
    )

    recall_value = recall_score(
        labels,
        pred,
        zero_division=0,
    )

    f1 = f1_score(
        labels,
        pred,
        zero_division=0,
    )

    print(
        f"{expert_name}: "
        f"AUC={auc_score:.4f}, "
        f"AUPR={aupr:.4f}, "
        f"Accuracy={accuracy:.4f}, "
        f"Precision={precision_value:.4f}, "
        f"Recall={recall_value:.4f}, "
        f"F1={f1:.4f}"
    )

    return [
        auc_score,
        aupr,
        accuracy,
        precision_value,
        recall_value,
        f1,
    ]


def train(data, args):
    set_seed(args.random_seed)

    args.miRNA_number = data["miRNA_number"]
    args.disease_number = data["disease_number"]

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    results_dir = os.path.join(
        "results",
        f"run_{timestamp}",
    )

    models_dir = os.path.join(
        "models",
        f"run_{timestamp}",
    )

    os.makedirs(
        results_dir,
        exist_ok=True,
    )

    os.makedirs(
        models_dir,
        exist_ok=True,
    )

    positives, negatives = (
        build_negative_pairs(
            data,
            args,
        )
    )

    if len(positives) < args.kfolds:
        raise ValueError(
            "Positive sample count is smaller than kfolds."
        )

    if len(negatives) < args.kfolds:
        raise ValueError(
            "Negative sample count is smaller than kfolds."
        )

    pos_kf = KFold(
        n_splits=args.kfolds,
        shuffle=True,
        random_state=args.random_seed,
    )

    neg_kf = KFold(
        n_splits=args.kfolds,
        shuffle=True,
        random_state=args.random_seed,
    )

    pos_splits = list(
        pos_kf.split(positives)
    )

    neg_splits = list(
        neg_kf.split(negatives)
    )

    all_score = []
    all_train_losses = []
    all_predictions = []

    save_epochs = sorted(
        set(
            e
            for e in (
                1,
                100,
                args.epochs,
            )
            if 1 <= e <= args.epochs
        )
    )

    for fold in range(args.kfolds):
        pos_train_idx, pos_test_idx = (
            pos_splits[fold]
        )

        neg_train_idx, neg_test_idx = (
            neg_splits[fold]
        )

        train_pos = positives[
            pos_train_idx
        ]

        test_pos = positives[
            pos_test_idx
        ]

        train_neg = negatives[
            neg_train_idx
        ]

        test_neg = negatives[
            neg_test_idx
        ]

        train_samples = make_samples(
            train_pos,
            train_neg,
            args.random_seed + fold,
        )

        test_samples = make_samples(
            test_pos,
            test_neg,
            args.random_seed + 1000 + fold,
        )

        (
            mm_graph,
            dd_graph,
            md_graph,
            miRNA_th,
            disease_th,
        ) = build_fold_data(
            data,
            train_pos,
            args,
        )

        model = MDA_MOE(
            args
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd,
        )

        criterion = nn.BCELoss()

        fold_losses = []

        epochs = trange(
            args.epochs,
            desc=f"Fold {fold + 1}",
        )

        for epoch in epochs:
            model.train()

            optimizer.zero_grad()

            train_score, _, _, _, _, _ = (
                model(
                    mm_graph,
                    dd_graph,
                    md_graph,
                    miRNA_th,
                    disease_th,
                    train_samples,
                )
            )

            labels = th.tensor(
                train_samples[:, 2],
                dtype=th.float32,
                device=device,
            )

            loss = criterion(
                th.flatten(
                    train_score
                ),
                labels,
            )

            loss.backward()

            optimizer.step()

            fold_losses.append(
                loss.item()
            )

            epochs.set_postfix(
                loss=f"{loss.item():.4f}"
            )

            if (
                epoch + 1
            ) in save_epochs:
                path = os.path.join(
                    models_dir,
                    (
                        f"model_epoch_"
                        f"{epoch + 1}_"
                        f"fold_{fold + 1}.pth"
                    ),
                )

                th.save(
                    model.state_dict(),
                    path,
                )

        all_train_losses.append(
            fold_losses
        )

        metrics, score = evaluate(
            model,
            mm_graph,
            dd_graph,
            md_graph,
            miRNA_th,
            disease_th,
            test_samples,
        )

        print(
            f"Fold {fold + 1}: "
            f"AUC={metrics[0]:.6f}, "
            f"AUPR={metrics[1]:.6f}, "
            f"Accuracy={metrics[2]:.6f}, "
            f"Precision={metrics[3]:.6f}, "
            f"Recall={metrics[4]:.6f}, "
            f"F1={metrics[5]:.6f}"
        )

        all_score.append(
            metrics
        )

        fold_predictions = np.concatenate(
            (
                test_samples.astype(
                    np.float64
                ),
                score.reshape(
                    -1,
                    1,
                ),
            ),
            axis=1,
        )

        all_predictions.append(
            fold_predictions
        )

        final_model_path = os.path.join(
            models_dir,
            f"final_model_fold_{fold + 1}.pth",
        )

        th.save(
            model.state_dict(),
            final_model_path,
        )

    all_score = np.asarray(
        all_score,
        dtype=np.float64,
    )

    cv_mean = np.mean(
        all_score,
        axis=0,
    )

    cv_std = np.std(
        all_score,
        axis=0,
        ddof=1,
    )

    print(
        f"{args.kfolds}-Fold Result"
    )

    print(
        "AUC: %.4f ± %.4f"
        % (
            cv_mean[0],
            cv_std[0],
        )
    )

    print(
        "AUPR: %.4f ± %.4f"
        % (
            cv_mean[1],
            cv_std[1],
        )
    )

    print(
        "Accuracy: %.4f ± %.4f"
        % (
            cv_mean[2],
            cv_std[2],
        )
    )

    print(
        "Precision: %.4f ± %.4f"
        % (
            cv_mean[3],
            cv_std[3],
        )
    )

    print(
        "Recall: %.4f ± %.4f"
        % (
            cv_mean[4],
            cv_std[4],
        )
    )

    print(
        "F1: %.4f ± %.4f"
        % (
            cv_mean[5],
            cv_std[5],
        )
    )

    np.save(
        os.path.join(
            results_dir,
            "all_fold_metrics.npy",
        ),
        all_score,
    )

    np.save(
        os.path.join(
            results_dir,
            "all_fold_predictions.npy",
        ),
        np.array(
            all_predictions,
            dtype=object,
        ),
        allow_pickle=True,
    )

    plt.figure(
        figsize=(10, 6)
    )

    for fold, losses in enumerate(
        all_train_losses
    ):
        plt.plot(
            losses,
            label=f"Fold {fold + 1}",
        )

    plt.title(
        "Training Loss Across Folds"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            results_dir,
            "training_loss_all_folds.png",
        ),
        dpi=300,
    )

    plt.close()

    return all_score


if __name__ == "__main__":
    args = parse_args()

    print("Loading data...")

    data = get_data(
        args
    )

    print("Training model...")

    train(
        data,
        args,
    )
