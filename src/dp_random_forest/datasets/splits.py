import numpy as np
from scipy.sparse import issparse, vstack as sparse_vstack
from sklearn.model_selection import train_test_split


def train_val_test_split(
    X,
    y,
    *extras,
    test_size=0.1,
    val_size=0.1,
    random_state=42,
    stratify=True,
):
    """
    Deterministic three-way split. The train/test boundary matches a single
    sklearn train_test_split(..., test_size=test_size, random_state=random_state).
    Validation is carved from the training portion with a second split
    (random_state + 1). val_size is the fraction of the train+val pool that
    goes to validation.
    """
    strat = np.asarray(y) if stratify else None
    arrays = (X, y) + extras
    out1 = train_test_split(
        *arrays,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    n = len(arrays)
    train_parts = [out1[2 * i] for i in range(n)]
    test_parts = [out1[2 * i + 1] for i in range(n)]

    X_tv, y_tv = train_parts[0], train_parts[1]
    ext_tv = train_parts[2:]
    ext_te = test_parts[2:]

    y_tv_arr = np.asarray(y_tv)
    if stratify:
        _, counts = np.unique(y_tv_arr, return_counts=True)
        strat2 = y_tv_arr if np.all(counts >= 2) else None
    else:
        strat2 = None

    arrays_tv = (X_tv, y_tv) + tuple(ext_tv)
    out2 = train_test_split(
        *arrays_tv,
        test_size=val_size,
        random_state=random_state + 1,
        stratify=strat2,
    )
    train2 = [out2[2 * i] for i in range(n)]
    val2 = [out2[2 * i + 1] for i in range(n)]

    X_tr, y_tr = train2[0], train2[1]
    X_va, y_va = val2[0], val2[1]
    ext_tr = train2[2:]
    ext_va = val2[2:]

    result = {
        "X_train": X_tr,
        "X_val": X_va,
        "X_test": test_parts[0],
        "y_train": y_tr,
        "y_val": y_va,
        "y_test": test_parts[1],
    }
    if extras:
        result["group_train"] = ext_tr[0]
        result["group_val"] = ext_va[0]
        result["group_test"] = ext_te[0]
    return result


def concat_train_val(data):
    """Stack train and validation for a final fit (test remains held out)."""
    X = data["X_train"]
    y = np.asarray(data["y_train"])
    if "X_val" not in data:
        return X, y
    Xv = data["X_val"]
    if Xv.shape[0] == 0:
        return X, y
    yv = np.asarray(data["y_val"])
    if issparse(X) or issparse(Xv):
        return sparse_vstack((X, Xv), format="csr"), np.concatenate((y, yv))
    X = np.asarray(X)
    Xv = np.asarray(Xv)
    return np.vstack((X, Xv)), np.concatenate((y, yv))
