import torch
from threading import Lock

trace_lock = Lock()

traces: dict[str, dict[int, dict[str, torch.Tensor]]] = {}

