from .partition import Partition, MPI
from typing import Mapping, Tuple, Union, Callable
import sys
import numpy as np

TreeDataT = Mapping[str, np.ndarray]


def distribute(
    partition: Partition,
    data: TreeDataT,
    xyz_keys: Tuple[str, str, str],
    *,
    verbose: Union[bool, int] = False,
    verify_count: bool = True,
) -> TreeDataT:
    """Distribute data among MPI ranks according to data position and volume partition

    The position of each TreeData element is given by the x, y, and z columns
    specified with `xyz_keys`.

    Parameters
    ----------

    partition:
        The MPI partition defining which rank should own which subvolume of the
        data

    data:
        The treenode / coretree data that should be distributed

    xyz_keys:
        The columns in `data` that define the position of the object

    verbose:
        If True, print summary statistics of the distribute. If > 1, print
        statistics of each rank (i.e. how much data each rank sends to every
        other rank).

    verify_count:
        If True, make sure that total number of objects is conserved

    Returns
    -------
    data: TreeDataT
        The distributed treenode / coretree data (i.e. the data that this rank
        owns)

    """
    # get some MPI and partition parameters
    nranks = partition.nranks
    if nranks == 1:
        return data

    rank = partition.rank
    comm = partition.comm
    ranklist = np.array(partition.ranklist)

    # count number of particles we have
    total_to_send = len(data[xyz_keys[0]])

    if total_to_send > 0:
        # Check validity of coordinates
        for i in range(3):
            _x = data[xyz_keys[i]]
            _min = _x.min()
            _max = _x.max()
            if _min < 0 or _max > partition.box_size:
                print(
                    f"Error in distribute: position {xyz_keys[i]} out of range: [{_min}, {_max}]",
                    file=sys.stderr,
                    flush=True,
                )
                comm.Abort()

        # Find home of each particle
        _i = (data[xyz_keys[0]] / partition.extent[0]).astype(np.int32)
        _j = (data[xyz_keys[1]] / partition.extent[1]).astype(np.int32)
        _k = (data[xyz_keys[2]] / partition.extent[2]).astype(np.int32)

        _i = np.clip(_i, 0, partition.decomp[0] - 1)
        _j = np.clip(_j, 0, partition.decomp[1] - 1)
        _k = np.clip(_k, 0, partition.decomp[2] - 1)
        home_idx = ranklist[_i, _j, _k]
    else:
        home_idx = np.empty(0, dtype=np.int32)

    # sort by rank
    s = np.argsort(home_idx)
    home_idx = home_idx[s]

    # offsets and counts
    send_displacements = np.searchsorted(home_idx, np.arange(nranks))
    send_displacements = send_displacements.astype(np.int32)
    send_counts = np.append(send_displacements[1:], total_to_send) - send_displacements
    send_counts = send_counts.astype(np.int32)

    # announce to each rank how many objects will be sent
    recv_counts = np.empty_like(send_counts)
    comm.Alltoall(send_counts, recv_counts)
    recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)

    # number of objects that this rank will receive
    total_to_receive = np.sum(recv_counts)

    # debug message
    if verbose > 1:
        for i in range(nranks):
            if rank == i:
                print(f"Distribute Debug Rank {i}")
                print(f" - rank has {total_to_send} particles")
                print(f" - rank receives {total_to_receive} particles")
                print(f" - send_counts:        {send_counts}")
                print(f" - send_displacements: {send_displacements}")
                print(f" - recv_counts:        {recv_counts}")
                print(f" - recv_displacements: {recv_displacements}")
                print(f"", flush=True)
            comm.Barrier()

    # send data all-to-all, each array individually
    data_new = {k: np.empty(total_to_receive, dtype=data[k].dtype) for k in data.keys()}

    for k in data.keys():
        d = data[k][s]
        s_msg = [d, (send_counts, send_displacements), d.dtype.char]
        r_msg = [data_new[k], (recv_counts, recv_displacements), d.dtype.char]
        comm.Alltoallv(s_msg, r_msg)

    if verify_count:
        local_counts = np.array(
            [len(data[xyz_keys[0]]), len(data_new[xyz_keys[0]])], dtype=np.int64
        )
        global_counts = np.empty_like(local_counts)
        comm.Reduce(local_counts, global_counts, op=MPI.SUM, root=0)
        if rank == 0 and global_counts[0] != global_counts[1]:
            print(
                f"Error in distribute: particle count during distribute was not maintained ({global_counts[0]} -> {global_counts[1]})",
                file=sys.stderr,
                flush=True,
            )
            comm.Abort()

    return data_new


def overload(
    partition: Partition,
    data: TreeDataT,
    overload_length: float,
    xyz_keys: Tuple[str, str, str],
    *,
    verbose: Union[bool, int] = False,
):
    """Copy data within an overload length to the 26 neighboring ranks

    This method assumes that the volume cube is periodic and will wrap the data
    around the boundary interfaces.

    Parameters
    ----------
    partition:
        The MPI partition defining which rank should own which subvolume of the
        data

    data:
        The treenode / coretree data that should be distributed

    overload_length:
        The thickness of the boundary layer that will be copied to the
        neighboring rank

    xyz_keys:
        The columns in `data` that define the position of the object

    verbose:
        If True, print summary statistics of the distribute. If > 1, print
        statistics of each rank (i.e. how much data each rank sends to every
        other rank).

    Returns
    -------
    data: TreeDataT
        The combined data of objects within the rank's subvolume as well as the
        objects within the overload region of neighboring ranks

    Notes
    -----

    The function does not change the objects' coordinates or alter any data.
    Objects that have been overloaded accross the periodic boundaries will still
    have the original positions. In case "local" coordinates are required, this
    will need to be done manually after calling this function.

    """
    nranks = partition.nranks
    if nranks == 1:
        return data

    rank = partition.rank
    comm = partition.comm

    neighbors = partition.neighbors

    # Find all overload regions each particle should be in
    overload = {}
    for (
        i,
        x,
    ) in enumerate(xyz_keys):
        _i = np.zeros_like(data[x], dtype=np.int8)
        _i[data[x] < partition.origin[i] + overload_length] = -1
        _i[data[x] > partition.origin[i] + partition.extent[i] - overload_length] = 1
        overload[i] = _i

    # Get particle indices of each of the 27 neighbors overload
    exchange_indices = [np.empty(0, dtype=np.int64)] * nranks

    def add_exchange_indices(mask, i, j, k):
        n = neighbors[i + 1, j + 1, k + 1]
        if n != rank:
            exchange_indices[n] = np.union1d(exchange_indices[n], np.nonzero(mask)[0])

    for i in [-1, 1]:
        # face
        maski = overload[0] == i
        add_exchange_indices(maski, i, 0, 0)

        for j in [-1, 1]:
            # edge
            maskj = maski & (overload[1] == j)
            add_exchange_indices(maskj, i, j, 0)

            for k in [-1, 1]:
                # corner
                maskk = maskj & (overload[2] == k)
                add_exchange_indices(maskk, i, j, k)

        for k in [-1, 1]:
            # edge
            maskk = maski & (overload[2] == k)
            add_exchange_indices(maskk, i, 0, k)

    for j in [-1, 1]:
        # face
        maskj = overload[1] == j
        add_exchange_indices(maskj, 0, j, 0)

        for k in [-1, 1]:
            # edge
            maskk = maskj & (overload[2] == k)
            add_exchange_indices(maskk, 0, j, k)

    for k in [-1, 1]:
        # face
        maskk = overload[2] == k
        add_exchange_indices(maskk, 0, 0, k)

    # Check how many elements will be sent
    send_counts = np.array([len(i) for i in exchange_indices], dtype=np.int32)
    send_idx = np.concatenate(exchange_indices)
    send_displacements = np.insert(np.cumsum(send_counts)[:-1], 0, 0)
    total_to_send = np.sum(send_counts)

    # Check how many elements will be received
    recv_counts = np.empty_like(send_counts)
    comm.Alltoall(send_counts, recv_counts)
    recv_displacements = np.insert(np.cumsum(recv_counts)[:-1], 0, 0)
    total_to_receive = np.sum(recv_counts)

    # debug message
    if verbose > 1:
        for i in range(nranks):
            if rank == i:
                print(f"Overload Debug Rank {i}")
                print(f" - rank sends    {total_to_send:10d} particles")
                print(f" - rank receives {total_to_receive:10d} particles")
                print(f" - send_counts:        {send_counts}")
                print(f" - send_displacements: {send_displacements}")
                print(f" - recv_counts:        {recv_counts}")
                print(f" - recv_displacements: {recv_displacements}")
                print(f" - overload_x: {overload[0]}")
                print(f" - overload_y: {overload[1]}")
                print(f" - overload_z: {overload[2]}")
                print(f" - send_idx: {send_idx}")
                print(f"", flush=True)
            comm.Barrier()

    # send data all-to-all, each array individually
    data_new = {}

    for k in data.keys():
        # prepare send-array
        ds = data[k][send_idx]
        # prepare recv-array
        dr = np.empty(total_to_receive, dtype=ds.dtype)
        # exchange data
        s_msg = [ds, (send_counts, send_displacements), ds.dtype.char]
        r_msg = [dr, (recv_counts, recv_displacements), ds.dtype.char]
        comm.Alltoallv(s_msg, r_msg)
        # add received data to original data
        data_new[k] = np.concatenate((data[k], dr))

    return data_new


def exchange(
    partition: Partition,
    data: dict,
    key: str,
    local_keys: np.ndarray,
    *,
    verbose: bool = False,
    filter_key: Union[int, Callable[[np.ndarray], np.ndarray]] = None,
    do_all2all: bool = False,
    replace_notfound_key: int = None,
):
    """Distribute data among neighboring ranks and all2all by a key

    This function will assign data to the rank that owns the key. The keys that the local rank owns are given by
    ``local_keys``, which should be unique. The keys of the data that the local rank currently has is in ``data[key]``.
    Certain values can be ignored by setting filter_key to that value or by setting filter_key to a (vectorized) function
    that returns ``True`` for keys that should be redistributed and ``False`` for keys that should be ignored.

    Parameters
    ----------
    partition:
        The MPI partition defining which rank should own which subvolume of the
        data

    data:
        The treenode / coretree data that should be distributed

    key:
        The name of the column that contains the key values, indicating which rank they
        should belong to

    local_keys:
        the key-values that this rank should own

    verbose:
        If True, print summary statistics of the exchange. If > 1, print
        statistics of each rank (i.e. how much data each rank sends to every
        other rank).

    filter_key:
        value which keys should be ignored (and therefore will not be moved) or function
        that applied to the key column returns a boolean array containing ``True``
        (=exchange) and ``False`` (=ignore)

    do_all2all:
        if ``False``, will first exchange objects with the 26 neighboring ranks, and if
        there are keys that could not be assigned among the neighbors, those will be
        distributed all2all. If ``True``, will do an all2all exchange from the beginning

    replace_notfound_key:
        for objects that could not be assigned, the "key" value will be set to this value.
        If ``None``, don't do anything.


    Returns
    -------
    data: TreeDataT
        All objects that belong to this rank, given by ``local_keys``

    Notes
    -----
    This function is designed for the usecase that most of the objects are already on
    the correct rank. If that's not the case, performance might be poor.

    """
    comm = partition.comm
    rank = partition.rank
    nranks = partition.nranks
    if nranks == 1:
        return data

    if do_all2all:
        # exchange particles with all ranks
        exchange_comm = comm
        exchange_nranks = nranks
        exchange_Alltoall = exchange_comm.Alltoall
        exchange_Alltoallv = exchange_comm.Alltoallv
        exchange_Allgather = exchange_comm.Allgather
        exchange_Allgatherv = exchange_comm.Allgatherv

    else:
        # exchange particles with the 26 neighboring ranks
        exchange_comm = partition.comm26
        exchange_nranks = partition.neighbors26_count
        exchange_Alltoall = exchange_comm.Neighbor_alltoall
        exchange_Alltoallv = exchange_comm.Neighbor_alltoallv
        exchange_Allgather = exchange_comm.Neighbor_allgather
        exchange_Allgatherv = exchange_comm.Neighbor_allgatherv

    localcount = len(data[key])
    data_keys = np.unique(data[key])
    if filter_key is not None:
        if callable(filter_key):
            data_keys = data_keys[filter_key(data_keys)]
        else:
            data_keys = data_keys[data_keys != filter_key]

    # find local matches
    islocal = np.isin(data_keys, local_keys, assume_unique=True)
    nonlocal_data = data_keys[~islocal]

    # communicate nonlocal descendants
    local_orphan_count = np.array([len(nonlocal_data)], dtype=np.int32)
    orphan_counts = np.empty(exchange_nranks, dtype=np.int32)
    exchange_Allgather(local_orphan_count, orphan_counts)
    total_orphan_count = np.sum(orphan_counts)
    orphan_offsets = np.insert(np.cumsum(orphan_counts)[:-1], 0, 0)
    orphan_data = np.empty(total_orphan_count, dtype=nonlocal_data.dtype)
    orphan_ranks = np.empty(total_orphan_count, dtype=np.int32)
    for i in range(exchange_nranks):
        low = orphan_offsets[i]
        high = low + orphan_counts[i]
        orphan_ranks[low:high] = i
    exchange_Allgatherv(
        nonlocal_data,
        [orphan_data, (orphan_counts, orphan_offsets), nonlocal_data.dtype.char],
    )

    if verbose > 1:
        for i in range(nranks):
            if rank == i:
                print(f"Debug Desc Exchange (nonlocal), rank {i}")
                print(f" - send {local_orphan_count}")
                print(f" - recv {orphan_counts}")
                print(f"", flush=True)
            comm.Barrier()

    # check if we have any of them
    orphan_islocal = np.isin(orphan_data, local_keys)

    # ask
    orphan_requests_send = orphan_data[orphan_islocal]
    orphan_requests_send_ranks = orphan_ranks[orphan_islocal]
    orphan_requests_send_offsets = np.searchsorted(
        orphan_requests_send_ranks, np.arange(exchange_nranks)
    )
    orphan_requests_send_counts = (
        np.append(orphan_requests_send_offsets[1:], len(orphan_requests_send))
        - orphan_requests_send_offsets
    )
    orphan_requests_recv_counts = np.empty_like(orphan_requests_send_counts)
    exchange_Alltoall(orphan_requests_send_counts, orphan_requests_recv_counts)
    orphan_requests_recv_total = np.sum(orphan_requests_recv_counts)
    orphan_requests_recv_offsets = np.insert(
        np.cumsum(orphan_requests_recv_counts)[:-1], 0, 0
    )
    orphan_requests_recv = np.empty(
        orphan_requests_recv_total, dtype=orphan_requests_send.dtype
    )
    exchange_Alltoallv(
        [
            orphan_requests_send,
            (orphan_requests_send_counts, orphan_requests_send_offsets),
            orphan_requests_send.dtype.char,
        ],
        [
            orphan_requests_recv,
            (orphan_requests_recv_counts, orphan_requests_recv_offsets),
            orphan_requests_recv.dtype.char,
        ],
    )

    if verbose > 1:
        for i in range(nranks):
            if rank == i:
                print(f"Debug Desc Exchange (request), rank {i}")
                print(f" - will request {np.sum(orphan_islocal)} in total")
                print(f" - send req {orphan_requests_send_counts}")
                print(f" - recv req {orphan_requests_recv_counts}")
                print(f"", flush=True)
            comm.Barrier()

    # verify that we don't aks ourselves for particles
    if do_all2all and (
        orphan_requests_send_counts[rank] != 0 or orphan_requests_recv_counts[rank] != 0
    ):
        print(
            f"Error in exchange: rank {rank} is asking itself for an orphan halo: {orphan_requests_send_counts[rank]}/{orphan_requests_recv_counts[rank]}",
            file=sys.stderr,
            flush=True,
        )
        comm.Abort()

    # prepare data to send
    orphan_requests_indices = []
    orphan_requests_mask = np.zeros(localcount, dtype=np.bool)
    for i in range(exchange_nranks):
        req = orphan_requests_recv[
            orphan_requests_recv_offsets[i] : orphan_requests_recv_offsets[i]
            + orphan_requests_recv_counts[i]
        ]
        mask = np.isin(data[key], req)
        orphan_requests_indices.append(np.nonzero(mask)[0])
        orphan_requests_mask |= mask
    orphan_requests_send_counts = np.array(
        [len(i) for i in orphan_requests_indices], dtype=np.int32
    )
    orphan_requests_recv_counts = np.empty_like(orphan_requests_send_counts)
    exchange_Alltoall(orphan_requests_send_counts, orphan_requests_recv_counts)
    orphan_requests_recv_total = np.sum(orphan_requests_recv_counts)
    orphan_requests_send_offsets = np.insert(
        np.cumsum(orphan_requests_send_counts)[:-1], 0, 0
    )
    orphan_requests_recv_offsets = np.insert(
        np.cumsum(orphan_requests_recv_counts)[:-1], 0, 0
    )
    orphan_requests_indices = np.concatenate(orphan_requests_indices)

    if verbose > 1:
        for i in range(nranks):
            if rank == i:
                print(f"Debug Desc Exchange (to exchange), rank {i}")
                print(f" - send {orphan_requests_send_counts}")
                print(f" - recv {orphan_requests_recv_counts}")
                print(f"", flush=True)
            comm.Barrier()

    data_new = {}
    for k in data.keys():
        orphan_requests_send = data[k][orphan_requests_indices]
        orphan_requests_recv = np.empty(orphan_requests_recv_total, dtype=data[k].dtype)
        exchange_Alltoallv(
            [
                orphan_requests_send,
                (orphan_requests_send_counts, orphan_requests_send_offsets),
                orphan_requests_send.dtype.char,
            ],
            [
                orphan_requests_recv,
                (orphan_requests_recv_counts, orphan_requests_recv_offsets),
                orphan_requests_recv.dtype.char,
            ],
        )
        data_new[k] = np.concatenate(
            (data[k][~orphan_requests_mask], orphan_requests_recv)
        )

    if verbose > 1 and rank == 0:
        print("Exchange succeeded, verifying data integrity", flush=True)
    # comm.Barrier()

    # Verification
    localcount_after = len(data_new[key])
    localcount_missmatch = local_orphan_count[0]
    # calculate new missmatch
    my_data = np.unique(data_new[key])
    my_data = my_data[my_data >= 0]
    islocal = np.isin(my_data, local_keys, assume_unique=True)
    missing_keys = my_data[~islocal]
    localcount_missmatch_after = len(missing_keys)
    localcounts = np.array(
        [
            localcount,
            localcount_after,
            localcount_missmatch,
            localcount_missmatch_after,
        ],
        dtype=np.int64,
    )
    totalcounts = np.empty_like(localcounts)
    comm.Allreduce(localcounts, totalcounts)
    (
        totalcount_before,
        totalcount_after,
        totalcount_missmatch,
        totalcount_missmatch_after,
    ) = totalcounts

    if verbose and rank == 0:
        print(f"exchange summary ({'all2all' if do_all2all else '26-neighbors'}):")
        print(
            f"   Ntot -> Ntot: {totalcount_before:10d} -> {totalcount_after:10d} (should remain the same)"
        )
        print(
            f"   Orph -> Orph: {totalcount_missmatch:10d} -> {totalcount_missmatch_after:10d} (should be 0 after)"
        )
        print("", flush=True)

    # did we conserve number of particles?
    if rank == 0 and totalcount_before != totalcount_after:
        print(
            f"Error in exchange: Lost halos during progenitor exchange: {totalcount_before} -> {totalcount_after}",
            file=sys.stderr,
            flush=True,
        )
        comm.Abort()

    # if we were not able to assign all orphans to the 26 neighbors, try all2all
    if not do_all2all and totalcount_missmatch_after > 0:
        if verbose and rank == 0:
            print(
                f"exchange all2all since neighbor exchange was not able to assign all: {totalcount_missmatch} -> {totalcount_missmatch_after}",
                flush=True,
            )
        return exchange(
            partition,
            data_new,
            key,
            local_keys,
            verbose=verbose,
            filter_key=filter_key,
            do_all2all=True,
            replace_notfound_key=replace_notfound_key,
        )

    # if we are still not able to assign all orphans, replace key or abort after printing some debug messages
    if replace_notfound_key is not None and localcount_missmatch_after > 0:
        d = data_new[key]
        d[np.isin(d, missing_keys)] = replace_notfound_key
    for i in range(nranks):
        if rank == i:
            if localcount_missmatch_after != 0:
                print(
                    f"Warning from rank {rank} in exchange: Unable to assign all progenitors to correct ranks (failed for {localcount_missmatch_after} out of {localcount_missmatch})"
                )
                print(f"Could not assign keys: ", missing_keys)
                print("", flush=True)
        comm.Barrier()

    if rank == 0 and totalcount_missmatch_after != 0:
        if replace_notfound_key is None:
            print(
                f"Error in exchange: Unable to assign all progenitors to correct ranks (tried to reassign {totalcount_missmatch}, failed for {totalcount_missmatch_after})",
                file=sys.stderr,
                flush=True,
            )
            comm.Abort()
        else:
            print(
                f"Warning in exchange: Unable to assign all progenitors to correct ranks (tried to reassign {totalcount_missmatch}, failed for {totalcount_missmatch_after}), replacing missing values with {replace_notfound_key}",
                flush=True,
            )

    return data_new
