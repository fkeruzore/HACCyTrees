import numpy as np
import pygio
import os, glob
from typing import Mapping, Sequence

from .partition import Partition
from .timer import Timer

class GenericIOStore:
    """A temporary storage that uses GenericIO to save data

    The data has to be a Structure-of-Arrays (SoA) (i.e. a python dictionary 
    `str`->`np.ndarray`), with each array having the same length.

    This class behaves like a python dictionary of `SoA`s. If `temporary_path` 
    is set, the arrays associated to a key will be stored in GenericIO files,
    otherwise, they will be kept in memory.

    Parameters
    ----------
    partition : :class:`haccytrees.utils.partition.Partition`
        A Partition instance defining the MPI layout
    
    temporary_path : str
        The base filesystem path where temporary data is stored. If `None`, the 
        data will be kept in memory.

    Methods
    -------
    __setitem__
    __getitem__
    get_field
    pop
    remove

    Examples
    --------

    >>> # Creating a partition
    >>> partition = haccytools.utils.partition.Parition(1.0)
    >>> # Creating a store
    >>> store = GenericIOStore(partition, './tmp')
    >>> data = {x: np.random.uniform(10) for x in 'xyz'}
    >>> store['pos_0'] = data
    >>> del data
    >>> # Do some memory-expensive stuff until you need the data again...
    >>> data_0 = store['pos_0']
    >>> # Cleanup
    >>> store.remove('pos_0')

    """
    def __init__(self, partition: Partition, temporary_path: str=None):
        self._temporary_path = temporary_path
        self._data = {}
        self._partition = partition

    def __setitem__(self, key: str, data: Mapping[str, np.ndarray]) -> None:
        """Adding an SoA to the storage

        Parameters
        ----------
        key 
            The storage key, will be appended to the `temporary_path` and
            therefore has to be a valid if used in a filesystem path.
        data
            The SoA to be added to the store. A dictionary with types
            `{str: np.ndarray}`, where the numpy array have to have the same shape
            and `dim=1`.
        """
        if self._temporary_path is None:
            self._data[key] = data
        else:
            rl = self._partition.box_size
            with Timer("temp storage: writing (GIO)", None):
                pygio.write_genericio(f"{self._temporary_path}_{key}.tmp.gio", data, phys_origin=[0., 0., 0.], phys_scale=[rl, rl, rl])

    def __getitem__(self, key: str) -> Mapping[str, np.ndarray]:
        """Retrieve a Structure-of-Arrays from the store

        Parameters
        ----------
        key
            The storage key of the SoA

        Returns
        -------
        Mapping[str, np.ndarray]
            The SoA associated with the key. A python dictionary of type `{str: np.ndarray}`
        """
        if self._temporary_path is None:
            return self._data[key]
        else:
            with Timer("temp storage: reading (GIO)", None):
                return pygio.read_genericio(f"{self._temporary_path}_{key}.tmp.gio", redistribute=pygio.PyGenericIO.MismatchBehavior.MismatchDisallowed)

    def get_field(self, key: str, field:Sequence[str]) -> np.ndarray:
        """Retrieve specific arrays in a Structure-of-Arrays from the store

        Parameters
        ----------
        key
            The storage key of the SoA
        field
            The keys of the specific arrays that are to be returned. Can be a `str`
            or a `list` of `str`.

        Returns
        -------
        Mapping[str, np.ndarray]
            The SoA associated with the key, with only the fields specified. 
            A python dictionary of type `{str: np.ndarray}`.
        """
        if not isinstance(field, list):
            field = [field]
        if self._temporary_path is None:
            return {f: self._data[key][f] for f in field}
        else:
            with Timer("temp storage: reading (GIO)", None):
                return pygio.read_genericio(f"{self._temporary_path}_{key}.tmp.gio", field, redistribute=pygio.PyGenericIO.MismatchBehavior.MismatchDisallowed)

    def remove(self, key: str) -> None:
        """Delete a stored SoA (from memory or disk)

        Parameters
        ----------
        key
            The storage key of the SoA
        """
        if self._temporary_path is None:
            self._data.pop(key)
        else:
            self._partition.comm.Barrier()
            with Timer("temp storage: cleanup (GIO)", None):
                if self._partition.rank == 0:
                    for f in glob.glob(f"{self._temporary_path}_{key}.tmp.gio*"):
                        os.remove(f)
            self._partition.comm.Barrier()

    def pop(self, key: str) -> Mapping[str, np.ndarray]:
        """Retrieve a Structure-of-Arrays from the store and remove the SoA

        Parameters
        ----------
        key
            The storage key of the SoA

        Returns
        -------
        Mapping[str, np.ndarray]
            The SoA associated with the key. A python dictionary of type `{str: np.ndarray}`
        """
        if self._temporary_path is None:
            return self._data.pop(key)
        else:
            d = self[key]
            self.remove(key)
            return d


class NumpyStore:
    """A temporary storage that uses numpy.savez to save data

    The data has to be a dictionary of arrays (i.e. `str`->`np.ndarray`), 
    arrays can have variable lengths and dimensions. If `temporary_path` 
    is set, the arrays associated to a key will be stored in `.npz` files (one
    per MPI rank), otherwise, they will be kept in memory.

    :param partition: A Partition instance defining the MPI layout
    :type partition: :class:`haccytrees.utils.partition.Partition`
    
    :param temporary_path: The base filesystem path where temporary data is
        stored. If `None`, the data will be kept in memory.
    :type temporary_path: `Optional[str]`

    Examples
    --------

    >>> # Creating a partition
    >>> partition = haccytools.utils.partition.Parition(1.0)
    >>> # Creating a store
    >>> store = NumpyStore(partition, './tmp')
    >>> data = {x: np.random.uniform(10) for x in 'xyz'}
    >>> store['pos_0'] = data
    >>> # Do some memory-expensive stuff until you need the data again...
    >>> data_0 = store['pos_0']
    >>> # Cleanup
    >>> store.remove('pos_0')

    """
    def __init__(self, partition: Partition, temporary_path: str=None):
        self._temporary_path = temporary_path
        self._data = {}
        self._partition = partition
        self._filename_fct = lambda key: f"{self._temporary_path}_{key}.tmp.rank-{self._partition.rank}.npz"

    def __setitem__(self, key: str, data: Mapping[str, np.ndarray]) -> None:
        """Adding a dictionary of arrays to the storage

        Parameters
        ----------
        key : str
            The storage key, will be appended to the `temporary_path` and
            therefore has to be a valid if used in a filesystem path.
        data : Mapping[str, np.ndarray]
            The data to be added to the store. A dictionary with types
            `{str: np.ndarray}`, where the numpy array can have variable shape
            and dimensions.
        """
        if self._temporary_path is None:
            self._data[key] = data
        else:
            with Timer("temp storage: writing (NPY)", None):
                np.savez(self._filename_fct(key), **data)

    def __getitem__(self, key: str) -> Mapping[str, np.ndarray]:
        """Retrieve a dictionary of arrays from the store

        Parameters
        ----------
        key
            The storage key

        Returns
        -------
        Mapping[str, np.ndarray]
            The data associated with the key. A python dictionary of type `{str: np.ndarray}`
        """
        if self._temporary_path is None:
            return self._data[key]
        else:
            with Timer("temp storage: reading (NPY)", None):
                return np.load(self._filename_fct(key))

    def remove(self, key: str) -> None:
        """Delete stored data associated with key (from memory or disk)

        Parameters
        ----------
        key
            The storage key of the data
        """
        if self._temporary_path is None:
            self._data.pop(key)
        else:
            with Timer("temp storage: cleanup (NPY)", None):
                os.remove(self._filename_fct(key))
                   
    def pop(self, key: str) -> Mapping[str, np.ndarray]:
        """Retrieve stored data and remove from storage (memory or disk)

        Parameters
        ----------
        key
            The storage key of the data

        Returns
        -------
        Mapping[str, np.ndarray]
            The data associated with the key. A python dictionary of type `{str: np.ndarray}`
        """
        if self._temporary_path is None:
            return self._data.pop(key)
        else:
            d = self[key]
            self.remove(key)
            return d