import numpy as np
import numba
from typing import Mapping, List
from ..simulations import Simulation, Cosmology

# TODO: parallelize numba functions: act on each root individually
@numba.jit(nopython=True)
def _create_submass_hostidx(
    snapnum,
    desc_index,
    snap0,
    sm_mainhostidx,
    sm_infallidx,
    sm_snapnum,
    sm_hostmassidx,
    scratch_idx,
):
    """calculate the index of the top host halo and immediate host (could be a subhalo)
    for each subhalo array
    """
    nhalos = len(snapnum)

    # A temporary array to keep track of the hosts in the current (sub)tree
    snap_mainhost_idx = np.empty(snap0 + 1, dtype=np.int64)
    snap_mainhost_idx[:] = -1

    # A temporary array to keep track of the direct host for a halo
    snap_mainhost_row = np.empty(snap0 + 1, dtype=np.int64)
    snap_mainhost_row[:] = -1

    current_row = 0  # current row in the scratch array
    lastsnap = snapnum[0]  # previous snapshot number
    current_idx = 0  # current position in the subhalo arrays
    for i in range(nhalos):
        # a completely new tree
        if desc_index[i] < 0:
            # reset arrays
            snap_mainhost_idx[:] = -1
            current_row = 0
            snap_mainhost_row[:] = -1
        # we are at a new "row" in matrix-layout
        # don't do this when we start a new tree (elif)
        elif snapnum[i] >= lastsnap:
            current_row += 1
            directhost_row = snap_mainhost_row[snapnum[i] + 1]
            assert directhost_row >= 0
            # fill-in the subhalos
            for s in range(snap0, snapnum[i], -1):
                sm_snapnum[current_idx] = s
                sm_mainhostidx[current_idx] = snap_mainhost_idx[s]
                scratch_idx[current_row, s] = -current_idx - 1

                # last index of the halo before merge
                sm_infallidx[current_idx] = i

                # direct host
                sm_hostmassidx[current_idx] = scratch_idx[directhost_row, s]

                # increment position in subhalo array
                current_idx += 1

            # remove invalidated roots
            for s in range(lastsnap, snapnum[i]):
                snap_mainhost_idx[s] = -1
                snap_mainhost_row[s] = -1

        snap_mainhost_idx[snapnum[i]] = i
        snap_mainhost_row[snapnum[i]] = current_row
        scratch_idx[current_row, snapnum[i]] = i
        lastsnap = snapnum[i]

    assert current_idx == len(sm_mainhostidx)


@numba.jit(nopython=True)
def _submass_model(
    snapnum,
    mass,
    desc_index,
    snap0,
    t_lb,
    tau_sub,
    zeta,
    submass,
    sm_directmassidx,
):
    nhalos = len(mass)

    # the exponential factor
    exp_fac = -1 / zeta

    current_idx = 0
    lastsnap = snapnum[0]
    for i in range(nhalos):
        if snapnum[i] >= lastsnap and desc_index[i] != -1:
            # new branch
            current_mass = mass[i]
            s_infall = snapnum[i]
            t_lb_current = 0.5 * (t_lb[s_infall] + t_lb[s_infall + 1])
            branch_nsub = snap0 - snapnum[i]

            # walk "backwards" through subhalo branch (starting from point of merger)
            for j in range(branch_nsub):
                # starting at the first snapshot the halo is a subhalo
                s = snapnum[i] + 1 + j
                # current_idx+branch_nsub is the start of the next subhalo
                idx = current_idx + branch_nsub - (j + 1)

                # is it a real host or another subhalo?
                if sm_directmassidx[idx] >= 0:
                    # real host – take mass after merger for first step, else previous
                    lux = sm_directmassidx[idx] + 1 * (j != 0)
                    mass_host = mass[lux]
                else:
                    lux = -sm_directmassidx[idx] - 1
                    assert lux < idx
                    mass_host = submass[lux]
                assert mass_host > 0

                mass_fac = zeta * (current_mass / mass_host) ** zeta
                delta_t = t_lb_current - t_lb[s]
                tau = tau_sub[s]

                submass[idx] = current_mass * (1 + mass_fac * delta_t / tau) ** exp_fac

                # update time and mass
                t_lb_current = t_lb[s]
                current_mass = submass[idx]

            current_idx += branch_nsub
        lastsnap = snapnum[i]

    assert current_idx == len(sm_directmassidx)


@numba.jit(nopython=True)
def _count_sub_number(snapnum, snap0, desc_index):
    # Count number of subs we'll need to allocate
    nhalos = len(snapnum)
    lastsnap = snapnum[0]
    nsub = 0
    maxrows = 0
    for i in range(nhalos):
        if desc_index[i] == -1:
            current_row = 1
        elif snapnum[i] >= lastsnap:
            # we are at a new "row" in matrix-layout
            # everything from z=0 to that halo will be a sub
            nsub += snap0 - snapnum[i]
            current_row += 1
            maxrows = max(maxrows, current_row)
        lastsnap = snapnum[i]
    return nsub, maxrows


def _create_submass_data(mass, snapnum, snap0, desc_index, tau_sub, t_lb, zeta):
    # Count number of subs we'll need to allocate
    nsub, maxrows = _count_sub_number(snapnum, snap0, desc_index)

    # Allocate arrays
    submass = np.empty(nsub, dtype=np.float32)
    submass_mainhostidx = np.empty(nsub, dtype=np.int64)
    submass_mainhostidx[:] = -(1 << 60)
    submass_directhostidx = np.empty(nsub, dtype=np.int64)
    submass_directhostidx[:] = -(1 << 60)
    submass_infallidx = np.empty(nsub, dtype=np.int64)
    submass_snapnum = np.zeros(nsub, dtype=snapnum.dtype)

    # Scratch space
    scratch_idx = np.empty((maxrows, snap0 + 1), dtype=np.int64)
    scratch_idx[:] = -(1 << 60)

    # Fill hostidx
    _create_submass_hostidx(
        snapnum,
        desc_index,
        snap0,
        submass_mainhostidx,
        submass_infallidx,
        submass_snapnum,
        submass_directhostidx,
        scratch_idx,
    )
    assert np.sum(submass_mainhostidx < 0) == 0
    assert np.sum(submass_directhostidx == -(1 << 60)) == 0

    # Fill submass (in reverse)
    _submass_model(
        snapnum,
        mass,
        desc_index,
        snap0,
        t_lb,
        tau_sub,
        zeta,
        submass,
        submass_directhostidx,
    )

    return {
        "mass": submass,
        "hostidx": submass_mainhostidx,
        "direct_hostidx": submass_directhostidx,
        "infallidx": submass_infallidx,
        "snapnum": submass_snapnum,
    }


@numba.jit(nopython=True, parallel=True)
def _compute_fsub_stats(
    subdata_offset, subdata_size, tree_node_mass, sub_mass, fsubtot, fsubmax
):
    nhalos = len(tree_node_mass)
    for i in numba.prange(nhalos):
        start = subdata_offset[i]
        end = subdata_offset[i] + subdata_size[i]
        _fsubtot = 0
        _fsubmax = 0
        for j in range(start, end):
            _fsubtot += sub_mass[j]
            _fsubmax = max(_fsubmax, sub_mass[j])
        fsubtot[i] = _fsubtot / tree_node_mass[i]
        fsubmax[i] = _fsubmax / tree_node_mass[i]


@numba.jit(nopython=True)
def _subdata_hostidx_order(nhalos, submass_hostidx):
    nsub = len(submass_hostidx)

    # Count subhalos of each halo
    subdata_size = np.zeros(nhalos, dtype=np.int64)
    for i in range(nsub):
        subdata_size[submass_hostidx[i]] += 1
    # convert to offset in final array
    subdata_offset = np.empty(nhalos, dtype=np.int64)
    current_offset = 0
    for i in range(nhalos):
        subdata_offset[i] = current_offset
        current_offset += subdata_size[i]
    assert current_offset == nsub

    # get ordering of data
    subdata_localoffsets = np.zeros(nhalos, dtype=np.int64)
    subdata_s = np.empty(nsub, dtype=np.int64)
    subdata_s[:] = -1
    for i in range(nsub):
        hidx = submass_hostidx[i]
        subdata_s[i] = subdata_offset[hidx] + subdata_localoffsets[hidx]
        subdata_localoffsets[hidx] += 1

    return subdata_s, subdata_offset, subdata_size


def create_submass_data(
    forest,
    simulation,
    *,
    zeta: float = 0.1,
    A: float = 1.1,
    mass_threshold: float = None,
    compute_fsub_stats: bool = False,
):
    if isinstance(simulation, str):
        simulation = Simulation.simulations[simulation]
    cosmo = simulation.cosmo
    a = simulation.step2a(np.array(simulation.cosmotools_steps))
    H = cosmo.hubble_parameter(a)
    H0 = 100 * cosmo.h

    # Bryan-Norman overdensity
    Delta_vir = cosmo.virial_overdensity(a)
    Delta_vir_0 = cosmo.virial_overdensity(1)

    # dynamical time of halo, in Gyr/h
    tau_dyn = 1.628 * (Delta_vir / Delta_vir_0) ** -0.5 * (H / H0) ** -1
    # characteristic timescale of subhalo mass loss
    tau_sub = tau_dyn / A

    # TAU IMRAN
    # 1.628/A * ( delta_vir(z)/delta_vir(0) )**(-0.5) * E(z)**(-1)

    # lookback time for each snapshot (in Gyr/h)
    t_lb = cosmo.lookback_time(a) * cosmo.h

    # number of (host) halos we're dealing with
    nhalos = len(forest["snapnum"])

    # apply submass model
    subhalo_data = _create_submass_data(
        forest["tree_node_mass"],
        forest["snapnum"],
        snap0=len(simulation.cosmotools_steps) - 1,
        desc_index=forest["descendant_idx"],
        tau_sub=tau_sub,
        t_lb=t_lb,
        zeta=zeta,
    )

    # Apply mass threshold
    if mass_threshold is not None:
        mask = subhalo_data["mass"] > mass_threshold
        for k in subhalo_data.keys():
            subhalo_data[k] = subhalo_data[k][mask]

    # order subdata by host halo
    subdata_s, subdata_offset, subdata_size = _subdata_hostidx_order(
        nhalos, subhalo_data["hostidx"]
    )

    # invert order
    subdata_s = np.argsort(subdata_s)

    for k in subhalo_data.keys():
        subhalo_data[k] = subhalo_data[k][subdata_s]
    forest["subdata_offset"] = subdata_offset
    forest["subdata_size"] = subdata_size

    # fsub statistics
    if compute_fsub_stats:
        forest["fsubtot"] = np.zeros(nhalos, dtype=np.float32)
        forest["fsubmax"] = np.zeros(nhalos, dtype=np.float32)
        _compute_fsub_stats(
            forest["subdata_offset"],
            forest["subdata_size"],
            forest["tree_node_mass"],
            subhalo_data["mass"],
            forest["fsubtot"],
            forest["fsubmax"],
        )

    return subhalo_data
