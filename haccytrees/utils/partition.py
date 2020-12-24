from mpi4py import MPI
import numpy as np
import sys, time

_comm = MPI.COMM_WORLD
_rank = _comm.Get_rank()
_nranks = _comm.Get_size()

class Partition:
    """An MPI partition of a cubic volume

    Parameters
    ----------
    box_size : float
        The length of the cubic volume

    create_topo26 : boolean
        If `True`, an additional graph communicator will be initialized 
        connecting all 26 direct neighbors symmetrically

    mpi_waittime : float
        Time in seconds for which the initialization will wait, can fix certain
        MPI issues if ranks are not ready (e.g. `PG with index not found`)

    Examples
    --------
    
    Using Partition on 8 MPI ranks to split a periodic unit-cube

    >>> partition = Partition(1.0)
    >>> partition.rank
    0
    >>> partition.decomp
    np.ndarray([2, 2, 2])
    >>> partition.coordinates
    np.ndarray([0, 0, 0])
    >>> partition.origin
    np.ndarray([0., 0., 0.])
    >>> partition.extent
    np.ndarray([0.5, 0.5, 0.5])


    """
    def __init__(self, box_size: float, create_topo26: bool=False, mpi_waittime: float=0):
        self._box_size = box_size
        self._rank = _rank
        self._nranks = _nranks
        self._decomp = MPI.Compute_dims(_nranks, [0,0,0])
        periodic = [True, True, True]
        time.sleep(mpi_waittime)
        self._topo = _comm.Create_cart(self._decomp, periods=periodic)
        self._coords = list(self._topo.coords)
        time.sleep(mpi_waittime)
        self._neighbors = np.empty((3, 3, 3), dtype=np.int32)
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    coord = [
                        (self._coords[0]+i) % self._decomp[0], 
                        (self._coords[1]+j) % self._decomp[1], 
                        (self._coords[2]+k) % self._decomp[2]
                        ]
                    neigh = self._topo.Get_cart_rank(coord)
                    self._neighbors[i+1, j+1, k+1] = neigh
                    # self._neighbors.append(neigh)

        self._extent = [self._box_size / self._decomp[i] for i in range(3)]
        self._origin = [self._coords[i] * self._extent[i] for i in range(3)]
        
        # A graph topology linking all 26 neighbors
        self._topo26 = None
        self._neighbors26 = None
        self._nneighbors26 = None
        if create_topo26:
            time.sleep(mpi_waittime)
            neighbors26 = np.unique(np.array([n for n in self._neighbors.flatten() if n != self._rank], dtype=np.int32))
            self._topo26 = self._topo.Create_dist_graph_adjacent(sources=neighbors26, destinations=neighbors26, reorder=False)
            assert(self._topo26.is_topo)
            inout_neighbors26 = self._topo26.inoutedges
            assert(len(inout_neighbors26[0]) == len(inout_neighbors26[1]))
            self._nneighbors26 = len(inout_neighbors26[0])
            for i in range(self._nneighbors26):
                if inout_neighbors26[0][i] != inout_neighbors26[1][i]:
                    print("topo 26: neighbors in sources and destinations are not ordered the same", file=sys.stderr, flush=True)
                    self._topo.Abort()
            self._neighbors26 = inout_neighbors26[0]

    def __del__(self):
        self._topo.Free()

    @property
    def box_size(self):
        """float: the length of the full cubic volume"""
        return self._box_size

    @property 
    def comm(self):
        """3D Cartesian MPI Topology / Communicator"""
        return self._topo

    @property
    def comm26(self):
        """Graph MPI Topology / Communicator, connecting the neighboring ranks
        (symmetric)"""
        return self._topo26

    @property
    def rank(self):
        """int: the MPI rank of this processor"""
        return self._topo.rank

    @property
    def nranks(self):
        """int: the total number of processors"""
        return self._nranks

    @property
    def decomp(self):
        """np.ndarray: the decomposition of the cubic volume: number of ranks along each dimension"""
        return self._decomp
        
    @property
    def coordinates(self):
        """np.ndarray: 3D indices of this processor"""
        return self._coords

    @property
    def extent(self):
        """np.ndarray: Length along each axis of this processors subvolume (same for all procs)"""
        return self._extent

    @property
    def origin(self) -> np.ndarray:
        """np.ndarray: Cartesian coordinates of the origin of this processor"""
        return self._origin


    def get_neighbor(self, dx: int, dy: int, dz: int) -> int:
        """get the rank of the neighbor at relative position (dx, dy, dz)
        
        Parameters
        ----------

        dx, dy, dz: int
            relative position, one of `[-1, 0, 1]`
        """
        return self._neighbors[dx+1, dy+1, dz+1]

    @property
    def neighbors(self):
        """np.ndarray: a 3x3x3 array with the ranks of the neighboring processes 
        (`neighbors[1,1,1]` is this processor)"""
        return self._neighbors

    @property
    def neighbors26(self):
        """np.ndarray: a flattened list of the unique neighboring ranks"""
        return self._neighbors26

    @property
    def neighbors26_count(self):
        """int: number of unique neighboring ranks"""
        return self._nneighbors26

    @property
    def ranklist(self):
        """np.ndarray: A complete list of ranks, aranged by their coordinates. 
        The array has shape `partition.decomp`"""
        ranklist = np.empty(self.decomp, dtype=np.int32)
        for i in range(self.decomp[0]):
            for j in range(self.decomp[1]):
                for k in range(self.decomp[2]):
                    ranklist[i, j, k] = self._topo.Get_cart_rank([i, j, k])
        return ranklist
