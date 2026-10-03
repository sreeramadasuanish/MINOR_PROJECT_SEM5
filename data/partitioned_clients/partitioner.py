"""
Partitioning logic for distributing MNIST across 3 clients with Non-IID label skews.
Simulates real-world heterogeneity across distributed nodes.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader


def extract_labels_from_dataset(dataset: Dataset) -> np.ndarray:
    """
    Extracts label vector from either torchvision Dataset or TensorDataset.
    """
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            return targets.cpu().numpy()
        return np.asarray(targets)
    elif hasattr(dataset, "tensors"):
        return dataset.tensors[1].cpu().numpy()
    else:
        # Fallback iteration
        return np.array([dataset[i][1] for i in range(len(dataset))])


def create_non_iid_partitions(
    train_dataset: Dataset,
    test_dataset: Dataset,
    num_clients: int = 3,
    skew_type: str = "structured",
    dirichlet_alpha: float = 0.5,
    seed: int = 42,
    max_train_samples_per_client: Optional[int] = None,
) -> Tuple[Dict[str, Subset], Dict[str, Subset], Dict[str, Dict[int, int]]]:
    """
    Partitions MNIST train and test sets across Client 1, Client 2, and Client 3.

    Skew Types:
        - 'structured':
            * Client 1: Digits 0, 1, 2, 3 dominant (75% local data)
            * Client 2: Digits 4, 5, 6 dominant (75% local data)
            * Client 3: Digits 7, 8, 9 dominant (75% local data)
        - 'dirichlet': Dirichlet distribution Dir(alpha) across all 10 classes
        - 'extreme': Pure pathological non-overlapping digit allocation

    Returns:
        client_train_subsets: Dict[client_id, Subset]
        client_test_subsets: Dict[client_id, Subset]
        class_distributions: Dict[client_id, Dict[class_idx, count]]
    """
    np.random.seed(seed)
    train_labels = extract_labels_from_dataset(train_dataset)
    test_labels = extract_labels_from_dataset(test_dataset)

    num_classes = 10
    client_ids = [f"Client_{i+1}" for i in range(num_clients)]

    # Group sample indices by class
    train_class_indices = {c: np.where(train_labels == c)[0] for c in range(num_classes)}
    test_class_indices = {c: np.where(test_labels == c)[0] for c in range(num_classes)}

    for c in range(num_classes):
        np.random.shuffle(train_class_indices[c])
        np.random.shuffle(test_class_indices[c])

    client_train_idx: Dict[str, List[int]] = {cid: [] for cid in client_ids}
    client_test_idx: Dict[str, List[int]] = {cid: [] for cid in client_ids}

    if skew_type == "structured":
        # Group digits into 3 specialty sets
        client_specialties = {
            "Client_1": [0, 1, 2, 3],
            "Client_2": [4, 5, 6],
            "Client_3": [7, 8, 9],
        }

        for c in range(num_classes):
            c_train_idx = train_class_indices[c]
            c_test_idx = test_class_indices[c]
            n_c_train = len(c_train_idx)
            n_c_test = len(c_test_idx)

            # Determine which client has primary affinity for class c
            primary_client = None
            for cid, digits in client_specialties.items():
                if c in digits:
                    primary_client = cid
                    break

            other_clients = [cid for cid in client_ids if cid != primary_client]

            # Primary client gets 70-80% of samples for this class, others share remainder
            primary_train_count = int(0.75 * n_c_train)
            other_train_count = (n_c_train - primary_train_count) // len(other_clients)

            primary_test_count = int(0.75 * n_c_test)
            other_test_count = (n_c_test - primary_test_count) // len(other_clients)

            # Train assignment
            client_train_idx[primary_client].extend(c_train_idx[:primary_train_count])
            offset = primary_train_count
            for oc in other_clients:
                client_train_idx[oc].extend(c_train_idx[offset : offset + other_train_count])
                offset += other_train_count

            # Test assignment
            client_test_idx[primary_client].extend(c_test_idx[:primary_test_count])
            offset_t = primary_test_count
            for oc in other_clients:
                client_test_idx[oc].extend(c_test_idx[offset_t : offset_t + other_test_count])
                offset_t += other_test_count

    elif skew_type == "dirichlet":
        # Dirichlet sampling per class
        for c in range(num_classes):
            c_train_idx = train_class_indices[c]
            c_test_idx = test_class_indices[c]
            
            proportions = np.random.dirichlet(np.repeat(dirichlet_alpha, num_clients))
            
            # Split train
            train_splits = (np.cumsum(proportions)[:-1] * len(c_train_idx)).astype(int)
            train_chunks = np.split(c_train_idx, train_splits)
            for i, cid in enumerate(client_ids):
                client_train_idx[cid].extend(train_chunks[i])

            # Split test
            test_splits = (np.cumsum(proportions)[:-1] * len(c_test_idx)).astype(int)
            test_chunks = np.split(c_test_idx, test_splits)
            for i, cid in enumerate(client_ids):
                client_test_idx[cid].extend(test_chunks[i])

    elif skew_type == "extreme":
        # Pure disjoint digit partition
        specialties = {
            "Client_1": [0, 1, 2],
            "Client_2": [3, 4, 5, 6],
            "Client_3": [7, 8, 9],
        }
        for cid, digits in specialties.items():
            for d in digits:
                client_train_idx[cid].extend(train_class_indices[d])
                client_test_idx[cid].extend(test_class_indices[d])

    # Subsample if maximum sample cap is specified
    if max_train_samples_per_client is not None:
        for cid in client_ids:
            if len(client_train_idx[cid]) > max_train_samples_per_client:
                perm = np.random.permutation(len(client_train_idx[cid]))
                client_train_idx[cid] = [client_train_idx[cid][i] for i in perm[:max_train_samples_per_client]]

    # Build Subsets and compute distribution statistics
    client_train_subsets: Dict[str, Subset] = {}
    client_test_subsets: Dict[str, Subset] = {}
    class_distributions: Dict[str, Dict[int, int]] = {}

    for cid in client_ids:
        # Shuffle indices
        np.random.shuffle(client_train_idx[cid])
        np.random.shuffle(client_test_idx[cid])

        client_train_subsets[cid] = Subset(train_dataset, client_train_idx[cid])
        client_test_subsets[cid] = Subset(test_dataset, client_test_idx[cid])

        # Class counts
        cid_labels = train_labels[client_train_idx[cid]]
        counts = {c: int(np.sum(cid_labels == c)) for c in range(num_classes)}
        class_distributions[cid] = counts

    return client_train_subsets, client_test_subsets, class_distributions


def compute_partition_distributions(class_distributions: Dict[str, Dict[int, int]]) -> List[Dict[str, Any]]:
    """
    Flattens distribution counts into tabular form suitable for Streamlit Altair charts.
    """
    data = []
    for client_id, counts in class_distributions.items():
        total = max(1, sum(counts.values()))
        for digit, cnt in counts.items():
            data.append({
                "Client": client_id,
                "Digit": str(digit),
                "Count": cnt,
                "Percentage": round((cnt / total) * 100, 2),
            })
    return data


def get_client_dataloaders(
    train_subsets: Dict[str, Subset],
    test_subsets: Dict[str, Subset],
    batch_size: int = 64,
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader]]:
    """
    Creates standard PyTorch DataLoaders for each client.
    """
    train_loaders = {
        cid: DataLoader(sub, batch_size=batch_size, shuffle=True)
        for cid, sub in train_subsets.items()
    }
    test_loaders = {
        cid: DataLoader(sub, batch_size=batch_size, shuffle=False)
        for cid, sub in test_subsets.items()
    }
    return train_loaders, test_loaders
