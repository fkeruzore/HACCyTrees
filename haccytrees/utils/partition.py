from mpi4py import MPI
import numpy as np
import sys, time

_comm = MPI.COMM_WORLD
_rank = _comm.Get_rank()
_nranks = _comm.Get_size()

class Partition:
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
        return self._box_size

    @property 
    def comm(self):
        return self._topo

    @property
    def comm26(self):
        return self._topo26

    @property
    def rank(self):
        return self._topo.rank

    @property
    def nranks(self):
        return self._nranks

    @property
    def decomp(self):
        return self._decomp
        
    @property
    def coordinates(self):
        return self._coords

    @property
    def extent(self):
        return self._extent

    @property
    def origin(self):
        return self._origin


    def get_neighbor(self, dx: int, dy: int, dz: int):
        return self._neighbors[dx+1, dy+1, dz+1]
        # return self._neighbors[((dx+1)*3 + (dy+1))*3 + (dz+1)]

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def neighbors26(self):
        return self._neighbors26

    @property
    def neighbors26_count(self):
        return self._nneighbors26

    @property
    def ranklist(self):
        ranklist = np.empty(self.decomp, dtype=np.int32)
        for i in range(self.decomp[0]):
            for j in range(self.decomp[1]):
                for k in range(self.decomp[2]):
                    ranklist[i, j, k] = self._topo.Get_cart_rank([i, j, k])
        return ranklist
