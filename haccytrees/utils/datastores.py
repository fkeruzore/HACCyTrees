import numpy as np
import pygio
import os, glob

from .partition import Partition
from .timer import Timer

class GenericIOStore:
    def __init__(self, partition: Partition, temporary_path: str=None):
        self._temporary_path = temporary_path
        self._data = {}
        self._partition = partition

    def __setitem__(self, key, data):
        if self._temporary_path is None:
            self._data[key] = data
        else:
            rl = self._partition.box_size
            with Timer("temp storage: writing (GIO)", None):
                pygio.write_genericio(f"{self._temporary_path}_{key}.tmp.gio", data, phys_origin=[0., 0., 0.], phys_scale=[rl, rl, rl])

    def __getitem__(self, key):
        if self._temporary_path is None:
            return self._data[key]
        else:
            with Timer("temp storage: reading (GIO)", None):
                return pygio.read_genericio(f"{self._temporary_path}_{key}.tmp.gio", redistribute=pygio.PyGenericIO.MismatchBehavior.MismatchDisallowed)

    def get_field(self, key, field):
        if not isinstance(field, list):
            field = [field]
        if self._temporary_path is None:
            return {f: self._data[key][f] for f in field}
        else:
            with Timer("temp storage: reading (GIO)", None):
                return pygio.read_genericio(f"{self._temporary_path}_{key}.tmp.gio", field, redistribute=pygio.PyGenericIO.MismatchBehavior.MismatchDisallowed)

    def remove(self, key):
        if self._temporary_path is None:
            self._data.pop(key)
        else:
            self._partition.comm.Barrier()
            with Timer("temp storage: cleanup (GIO)", None):
                if self._partition.rank == 0:
                    for f in glob.glob(f"{self._temporary_path}_{key}.tmp.gio*"):
                        os.remove(f)
            self._partition.comm.Barrier()

    def pop(self, key):
        if self._temporary_path is None:
            return self._data.pop(key)
        else:
            d = self[key]
            self.remove(key)
            return d

class NumpyStore:
    def __init__(self, partition: Partition, temporary_path: str=None):
        self._temporary_path = temporary_path
        self._data = {}
        self._partition = partition
        self._filename_fct = lambda key: f"{self._temporary_path}_{key}.tmp.rank-{self._partition.rank}.npz"

    def __setitem__(self, key, data):
        if self._temporary_path is None:
            self._data[key] = data
        else:
            with Timer("temp storage: writing (NPY)", None):
                np.savez(self._filename_fct(key), **data)

    def __getitem__(self, key):
        if self._temporary_path is None:
            return self._data[key]
        else:
            with Timer("temp storage: reading (NPY)", None):
                return np.load(self._filename_fct(key))

    def remove(self, key):
        if self._temporary_path is None:
            self._data.pop(key)
        else:
            with Timer("temp storage: cleanup (NPY)", None):
                os.remove(self._filename_fct(key))
                   
    def pop(self, key):
        if self._temporary_path is None:
            return self._data.pop(key)
        else:
            d = self[key]
            self.remove(key)
            return d