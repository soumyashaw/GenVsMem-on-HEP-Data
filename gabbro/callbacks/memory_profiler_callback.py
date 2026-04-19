import tracemalloc

import lightning as L
import torch.distributed as dist

from gabbro.utils.pylogger import get_pylogger


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0  # Default rank when not using distributed capabilities


class MemoryProfilerCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.pylogger = get_pylogger(__name__)
        tracemalloc.start()  # Start at the initialization or on_fit_start if you prefer

    def on_train_epoch_start(self, trainer, pl_module):
        self.snapshot_before = tracemalloc.take_snapshot()

    def on_train_epoch_end(self, trainer, pl_module):
        snapshot_after = tracemalloc.take_snapshot()
        self.display_top(self.snapshot_before, snapshot_after)
        self.snapshot_before = snapshot_after  # Update for next epoch

    def display_top(self, snapshot1, snapshot2, key_type="lineno", limit=10):
        """Display the top memory usage differences between two snapshots."""
        stats = snapshot2.compare_to(snapshot1, key_type=key_type)
        rank = get_rank()
        self.pylogger.info(f"\n[Rank {rank}] Top 10 memory usage differences for this epoch:")
        for stat in stats[:limit]:
            self.pylogger.info(f"[Rank {rank}] {stat}")
