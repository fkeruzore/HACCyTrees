import numpy as np
import psutil
from mpi4py import MPI

from mpipartition import Partition

# Memory debugging
def debug_memory(partition: Partition, msg: str):
    process = psutil.Process()
    localmem = np.array([process.memory_info().rss, process.memory_info().vms])
    globalmem_min = np.empty_like(localmem)
    globalmem_max = np.empty_like(localmem)
    partition.comm.Reduce(localmem, globalmem_min, op=MPI.MIN, root=0)
    partition.comm.Reduce(localmem, globalmem_max, op=MPI.MAX, root=0)

    if partition.rank == 0:
        print(f"MEMORY USAGE ({msg}): MIN / MAX (MB)")
        print(f"  RSS: {globalmem_min[0]//1024**2:6d} / {globalmem_max[0]//1024**2:6d}")
        print(f"  VMS: {globalmem_min[1]//1024**2:6d} / {globalmem_max[1]//1024**2:6d}")
        print("", flush=True)
