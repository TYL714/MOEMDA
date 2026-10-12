
    return gaussian


def _load_csv_matrix(path, index_col=None):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    matrix = pd.read_csv(path, index_col=index_col).values

    if matrix.ndim != 2:
        raise ValueError(f"Invalid matrix shape in {path}: {matrix.shape}")

    return matrix.astype(np.float32)


def _load_txt_matrix(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    matrix = np.loadtxt(path)

    if matrix.ndim != 2:
        raise ValueError(f"Invalid matrix shape in {path}: {matrix.shape}")

    return matrix.astype(np.float32)


def _validate_square_matrix(matrix, expected_size, matrix_name):
    if matrix.shape != (expected_size, expected_size):
        raise ValueError(
            f"{matrix_name} shape mismatch: "
            f"expected {(expected_size, expected_size)}, got {matrix.shape}"
        )


def _safe_weight_matrix(path, reference_matrix, loader):
    if not os.path.exists(path):
        return np.full_like(reference_matrix, 0.5, dtype=np.float32)

    try:
        weight = loader(path)
    except Exception:
        return np.full_like(reference_matrix, 0.5, dtype=np.float32)

    if weight.shape != reference_matrix.shape:
        return np.full_like(reference_matrix, 0.5, dtype=np.float32)

    return weight.astype(np.float32)


def get_data(args):
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(
            f"数据目录不存在: {args.data_dir}"
        )

    dataset_path = args.data_dir
    files = os.listdir(dataset_path)

    if any(f.endswith(".csv") for f in files):
        data = load_hmdd_v3_data(args, dataset_path)
    elif any(f.endswith(".txt") for f in files):
        data = load_hmdd_v32_data(args, dataset_path)
    else:
        raise ValueError(
            f"无法识别的数据集格式，目录内容: {files}"
        )

    args.miRNA_number = data["miRNA_number"]
    args.disease_number = data["disease_number"]

    print("成功加载数据集:")
    print(f"miRNA数量: {data['miRNA_number']}")
    print(f"疾病数量: {data['disease_number']}")
    print(f"已知关联数量: {len(data['md'])}")

    return data


def load_hmdd_v3_data(args, dataset_path):
    mirna_name_path = os.path.join(
        dataset_path,
        "miRNA name.csv",
    )
    disease_name_path = os.path.join(
        dataset_path,
        "disease number.csv",
    )
    md_path = os.path.join(
        dataset_path,
        "m_d.csv",
    )
    mf_path = os.path.join(
        dataset_path,
        "m_fs.csv",
    )
    dss_path = os.path.join(
        dataset_path,
        "d_ss.csv",
    )

    if not os.path.exists(mirna_name_path):
        raise FileNotFoundError(mirna_name_path)

    if not os.path.exists(disease_name_path):
        raise FileNotFoundError(disease_name_path)

    if not os.path.exists(md_path):
        raise FileNotFoundError(md_path)

    mirna_names = pd.read_csv(mirna_name_path)
    disease_names = pd.read_csv(disease_name_path)

    miRNA_number = len(mirna_names)
    disease_number = len(disease_names)

    md_df = pd.read_csv(md_path)
    md_matrix = md_df.values

    if md_matrix.ndim != 2:
        raise ValueError(
            f"m_d.csv shape invalid: {md_matrix.shape}"
        )

    if (
        md_matrix.shape[1] == disease_number + 1
        and md_matrix.shape[0] == miRNA_number
    ):
        md_matrix = md_matrix[:, 1:]

    if md_matrix.shape != (miRNA_number, disease_number):
        raise ValueError(
            "m_d.csv shape mismatch: "
            f"expected {(miRNA_number, disease_number)}, "
            f"got {md_matrix.shape}"
        )

    md_associations = np.argwhere(
        md_matrix == 1
    ).astype(np.int64)

    mf = _load_csv_matrix(
        mf_path,
        index_col=0,
    )
    dss = _load_csv_matrix(
        dss_path,
        index_col=0,
    )

    _validate_square_matrix(
        mf,
        miRNA_number,
        "miRNA functional similarity matrix",
    )
    _validate_square_matrix(
        dss,
        disease_number,
        "disease semantic similarity matrix",
