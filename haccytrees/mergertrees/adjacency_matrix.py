import numpy as np
import numba
from typing import Tuple, Mapping, List
from dataclasses import dataclass


@dataclass
class HaccyAdjGraph:
    A: np.ndarray  # adjacency matrix
    X: np.ndarray  # node features
    E: np.ndarray  # edge features
    L: np.ndarray  # node labels (tree_node_index)


@numba.jit(nopython=True)
def _fill_adj_bool(desc_index, a):
    nhalos = len(desc_index)
    for i in numba.prange(1, nhalos):
        a[i, desc_index[i]] = 1


@numba.jit(nopython=True)
def _fill_adj_mass(desc_index, mass, a):
    nhalos = len(desc_index)
    for i in numba.prange(1, nhalos):
        a[i, desc_index[i]] = mass[i]


def extract_adj_matrices(
    forest: Mapping[str, np.ndarray],
    simulation: str,
    target_idx: np.ndarray,
    *,
    with_mass_entries: bool = False,
    node_features: List[str] = None,
    edge_features: List[str] = None
) -> List[HaccyAdjGraph]:
    """Create adjacency matrix and feature matrices from (sub)trees

    Let's assume the subtree contains ``nhalos``. The adjacency matrix is a
    matrix of shape ``(nhalos, nhalos)`` with the entry ``(i, j)`` being
    non-zero if halo ``i`` merges into halo ``j``. The value can be set to the
    mass of halo ``i`` if ``with_mass_entries==True``, otherwise it will be
    ``1``.

    Parameters
    ----------
    forest the full haccytree forest

    simulation the simulation the forest is from

    target_idx a list/array of the root indices of the subtrees which shall be
        converted to an adjacency matrix

    with_mass_entries if ``True``, the connections in the adjacency matrix will
        contain the mass of the merging halo. If ``False``, matrix will only
        contain 0 and 1

    node_features a list of ``forest`` keys that will be included in the
        ``node_features`` matrix

    edge_features not implemented yet

    Returns
    -------
    adj_graphs: List[HaccyAdjGraph] a list of graph dataclasses containing the
        adjacency matrices and node and edge features

    """
    adj_graphs = []
    for idx in target_idx:
        nodes = forest['branch_size'][idx]
        start = idx
        end = idx+nodes
        if with_mass_entries:
            a = np.zeros((nodes, nodes), dtype=np.float32)
            _fill_adj_mass(forest['descendant_idx'][start:end]-start, 
                           forest['tree_node_mass'][start:end], 
                           a)
        else:
            a = np.zeros((nodes, nodes), dtype=np.uint8)
            _fill_adj_bool(forest['descendant_idx'][start:end]-start, a)

        if node_features is None:
            x = None
        else:
            x = np.hstack((forest[f][start:end] for f in node_features))

        if edge_features is None:
            e = None
        else:
            raise NotImplementedError("edge features not implemented yet")

        l = forest['tree_node_index'][start:end]

        g = HaccyAdjGraph(A=a, X=x, E=e, L=l)
        adj_graphs.append(g)

    return adj_graphs

