import numpy as np
import numba
from typing import Mapping, List
from ..simulations import Simulation, Cosmology

# TODO: parallelize numba functions: act on each root individually

@numba.jit(nopython=True)
def _create_submass_hostidx(snapnum, desc_index, snap0, submass_hostidx, submass_infallidx, submass_snapnum):
    nsub = len(submass_hostidx)
    nhalos = len(snapnum)
    # A temporary array to keep track of the hosts in the current (sub)tree
    snap_roots = np.empty(snap0+1, dtype=np.int64)
    snap_roots[:] = -1
    lastsnap = snapnum[0]
    current_idx = 0
    for i in range(nhalos):
        # print(i, snap_roots)
        if desc_index[i] < 0:
            # a completely new root
            snap_roots[:] = -1
        elif snapnum[i] >= lastsnap:
            # we are at a new "row" in matrix-layout, fill in the subhalos
            for s in range(snap0, snapnum[i], -1):
                submass_snapnum[current_idx] = s
                submass_hostidx[current_idx] = snap_roots[s]
                submass_infallidx[current_idx] = i
                current_idx += 1
            # remove invalidated roots
            for s in range(lastsnap, snapnum[i]):
                snap_roots[s] = -1
        snap_roots[snapnum[i]] = i
        lastsnap = snapnum[i]

    assert(current_idx == nsub)


@numba.jit(nopython=True)
def _create_submass_model(snapnum, mass, desc_index, snap0, t_lb, tau_sub, zeta, submass):
    nsub = len(submass)
    nhalos = len(mass)

    current_idx = nsub-1
    for i in range(nhalos-1, -1, -1):
        if desc_index[i] != -1 and desc_index[i] != i-1:
            # This halo merges
            s = snapnum[i]
            # the last halo merged between lastsnap and lastsnap+1
            tlb_infall = 0.5*(t_lb[s] + t_lb[s+1])
            # the mass of the host at the first timestep of the subhalo
            mass_host = mass[desc_index[i]]
            # the infall mass
            mass_infall = mass[i]
            # some factors in the mass model equation
            mass_fac = zeta * (mass_infall/mass_host)**zeta
            exp_fac = -1/zeta
            # loop over the snaps that need to be filled in
            for j in range(s+1, snap0+1):
                delta_t = tlb_infall - t_lb[j]
                tau = tau_sub[j]  # from the paper: tau is computed at t+delta_t
                submass[current_idx] = mass_infall * (1 + mass_fac*delta_t/tau)**exp_fac
                current_idx -= 1
                # assert(current_idx >= 0)
    assert(current_idx == -1)


@numba.jit(nopython=True)
def _create_submass_data(mass, snapnum, snap0, desc_index, tau_sub, t_lb, zeta):
    # Count number of subs we'll need to allocate
    nhalos = len(snapnum)
    lastsnap = snapnum[0]
    nsub = 0
    for i in range(nhalos):
        if snapnum[i] >= lastsnap and desc_index[i] >= 0:
            # we are at a new "row" in matrix-layout
            # everything from z=0 to that halo will be a sub
            nsub += snap0-snapnum[i]
        lastsnap = snapnum[i]

    # Allocate arrays
    submass = np.empty(nsub, dtype=np.float32)
    submass_hostidx = np.zeros(nsub, dtype=np.int64)
    submass_infallidx = np.zeros(nsub, dtype=np.int64)
    submass_snapnum = np.zeros(nsub, dtype=snapnum.dtype)

    # Fill hostidx
    _create_submass_hostidx(snapnum, desc_index, snap0, submass_hostidx, submass_infallidx, submass_snapnum)
    
    # Fill submass (in reverse)
    _create_submass_model(snapnum, mass, desc_index, snap0, t_lb, tau_sub, zeta, submass)

    return submass, submass_hostidx, submass_infallidx, submass_snapnum


@numba.jit(nopython=True, parallel=True)
def _compute_fsub_stats(subdata_offset, subdata_size, tree_node_mass, sub_mass, fsubtot, fsubmax):
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
    assert(current_offset == nsub)

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
    forest: Mapping[str, np.ndarray],
    simulation: Simulation,
    *,
    zeta: float = 0.1,
    A: float = 1.1,
    mass_threshold: float = None,
    compute_fsub_stats: bool = False
    ) -> Mapping[str, np.ndarray]:
    """Apply subhalo mass-modelling (Sultan+ 2020) to all merging halos

    This function will create a subhalo dataset containing the modelled masses
    as well as the forest indices to the host halo and the infall halo they
    belong to. It will also add a ``subdata_offset`` and ``subdata_size`` column
    to the forest data that can be used to look-up subhalos belonging to a halo.

    Parameters
    ----------
    forest
        the full merger tree forest data
    
    simulation
        the simulation string or Simulation instance

    zeta
        the zeta parameter in the Sultan+ 2020 mass model

    A
        the A parameter in the Sultan+ 2020 mass model

    mass_threshold
        if not None, the subhalo mass will only be modelled as long as it is
        above this threshold

    compute_fsub_stats
        if True, will calculate fsubtot (total mass of substructure) and
        fsubmax (mass of the largest substructure), where the numbers are 
        relative to the host-halo mass. These columns will be added to the
        ``forest`` data.

    Returns
    -------
    subhalo_data: Mapping[str, np.ndarray]
        the subhalo dataset, with the following columns:

        - ``mass``:      the modeled subhalo mass
        - ``hostidx``:   the array index to the host halo in the forest table
        - ``infallidx``: the array index to the infall halo in the forest table
        - ``snapnum``:   the snapshot number of the modeled subhalo mass
    """
    if isinstance(simulation, str):
        simulation = Simulation.simulations[simulation]
    cosmo = simulation.cosmo
    a = simulation.step2a(np.array(simulation.cosmotools_steps))
    H = cosmo.hubble_parameter(a)
    H0 = 100*cosmo.h

    # Bryan-Norman overdensity
    Delta_vir = cosmo.virial_overdensity(a)
    Delta_vir_0 = cosmo.virial_overdensity(1)

    # dynamical time of halo, in Gyr
    tau_dyn = 1.628 / cosmo.h * (Delta_vir/Delta_vir_0)**-0.5 * (H/H0)**-1
    # characteristic timescale of subhalo mass loss
    tau_sub = tau_dyn / A

    # lookback time for each snapshot
    t_lb = cosmo.lookback_time(a)

    # number of (host) halos we're dealing with
    nhalos = len(forest['snapnum'])

    # apply submass model
    sub_mass, sub_hostidx, sub_infallidx, sub_snapnum = _create_submass_data(
        forest['tree_node_mass'], 
        forest['snapnum'], 
        len(simulation.cosmotools_steps)-1,
        forest['descendant_idx'],
        tau_sub,
        t_lb, 
        zeta
        )

    subhalo_data = {
        'mass': sub_mass,
        'hostidx': sub_hostidx,
        'infallidx': sub_infallidx,
        'snapnum': sub_snapnum
    }

    # Apply mass threshold
    if mass_threshold is not None:
        mask = subhalo_data['mass'] > mass_threshold
        for k in subhalo_data.keys():
            subhalo_data[k] = subhalo_data[k][mask]

    # order subdata by host halo
    subdata_s, subdata_offset, subdata_size = _subdata_hostidx_order(nhalos, subhalo_data['hostidx'])
    for k in subhalo_data.keys():
        subhalo_data[k] = subhalo_data[k][subdata_s]
    forest['subdata_offset'] = subdata_offset
    forest['subdata_size'] = subdata_size

    # fsub statistics
    if compute_fsub_stats:
        forest['fsubtot'] = np.zeros(nhalos, dtype=np.float32)
        forest['fsubmax'] = np.zeros(nhalos, dtype=np.float32)
        _compute_fsub_stats(forest['subdata_offset'], forest['subdata_size'], forest['tree_node_mass'], subhalo_data['mass'], forest['fsubtot'], forest['fsubmax'])

    return subhalo_data
