"""Flower-free federated learning: FedAvg and FedProx over five hospital clients."""

from ml.federated.fedavg import fedavg_aggregate
from ml.federated.fedprox import copy_global_params
from ml.federated.server import run_federated
