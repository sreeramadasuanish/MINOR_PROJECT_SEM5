"""
Client data partitioning module for Non-IID skew simulation.
"""

from .partitioner import (
    create_non_iid_partitions,
    get_client_dataloaders,
    extract_labels_from_dataset,
    compute_partition_distributions,
)

__all__ = [
    "create_non_iid_partitions",
    "get_client_dataloaders",
    "extract_labels_from_dataset",
    "compute_partition_distributions",
]
