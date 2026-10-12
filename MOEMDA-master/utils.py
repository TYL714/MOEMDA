        md_path,
    ]

    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    mirna_list = []
    with open(
        mirna_number_path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                mirna_list.append(parts[1])

    disease_list = []
    with open(
        disease_number_path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                disease_list.append(parts[1])

    miRNA_number = len(mirna_list)
    disease_number = len(disease_list)

    mf = _load_txt_matrix(mf_path)
    dss = _load_txt_matrix(dss_path)

    _validate_square_matrix(
        mf,
        miRNA_number,
        "miRNA functional similarity matrix",
    )
    _validate_square_matrix(
        dss,
        disease_number,
        "disease semantic similarity matrix",
    )

    mfw_path = os.path.join(
        dataset_path,
        "miRNA functional similarity weight matrix.txt",
    )
    dsw_path = os.path.join(
        dataset_path,
        "disease semantic similarity weight matrix.txt",
    )

    mfw = _safe_weight_matrix(
        mfw_path,
        mf,
        _load_txt_matrix,
    )
    dsw = _safe_weight_matrix(
        dsw_path,
        dss,
        _load_txt_matrix,
    )

    md_associations = []

    with open(
        md_path,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            parts = line.strip().split("\t")

            if len(parts) < 2:
                continue

            i = int(parts[0]) - 1
            j = int(parts[1]) - 1

            if (
                0 <= i < miRNA_number
                and 0 <= j < disease_number
            ):
                md_associations.append([i, j])

    md_associations = np.asarray(
        md_associations,
        dtype=np.int64,
    )

    data = {
        "miRNA_number": miRNA_number,
        "disease_number": disease_number,
        "mf": mf,
        "dss": dss,
        "mfw": mfw,
        "dsw": dsw,
        "md": md_associations,
        "m_num": mirna_list,
        "d_num": disease_list,
    }

    return data
