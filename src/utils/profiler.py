"""
profiler.py
-----------
Phase 11 Publication Upgrade: Runtime Profiling Framework.

Provides CUDATimer and MemoryTracker for rigorous IEEE RA-L latency
and throughput evaluation. Includes synchronization for accurate GPU timing.
"""

import time
import torch
import logging

log = logging.getLogger(__name__)

class CUDATimer:
    """Rigorous CUDA timing with synchronization."""
    def __init__(self, name: str = "Timer", device: torch.device = None):
        self.name = name
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.start_event = None
        self.end_event = None
        self.start_time = 0.0
        self.is_cuda = self.device.type == "cuda"
        
        if self.is_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        if self.is_cuda:
            torch.cuda.synchronize()
            self.start_event.record()
        else:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_cuda:
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed_ms = self.start_event.elapsed_time(self.end_event)
        else:
            self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000.0
            
    def get_elapsed_ms(self) -> float:
        return self.elapsed_ms


class MemoryTracker:
    """Tracks peak GPU memory usage during a block."""
    def __init__(self, name: str = "MemoryTracker", device: torch.device = None):
        self.name = name
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_cuda = self.device.type == "cuda"
        
    def __enter__(self):
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_mem = torch.cuda.memory_allocated(self.device)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_cuda:
            self.end_mem = torch.cuda.memory_allocated(self.device)
            self.peak_mem = torch.cuda.max_memory_allocated(self.device)
        else:
            self.start_mem = 0
            self.end_mem = 0
            self.peak_mem = 0
            
    def get_peak_mb(self) -> float:
        return self.peak_mem / (1024 * 1024)
        
    def get_delta_mb(self) -> float:
        return (self.end_mem - self.start_mem) / (1024 * 1024)


class ThroughputTracker:
    """Tracks throughput (Hz) over multiple iterations."""
    def __init__(self):
        self.total_time_s = 0.0
        self.total_items = 0
        
    def update(self, time_ms: float, items: int = 1):
        self.total_time_s += time_ms / 1000.0
        self.total_items += items
        
    def get_throughput(self) -> float:
        if self.total_time_s == 0:
            return 0.0
        return self.total_items / self.total_time_s
