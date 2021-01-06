import numpy as np
import numba
from typing import Tuple, Mapping, List


def split_fragment_tag(tag: int) -> Tuple[int, int]:
    """Extracting the original fof_tag and fragment index from fragments

    Parameters
    ----------
    tag
        the fof_tag of the fragment, has to be negative by definition

    Returns
    -------
    fof_tag
        the original fof_tag of the FoF halo the fragment is associated with
    
    fragment_idx
        the index / enumeration of the fragment (0 == main fragment)

    Notes
    -----
    This function can also be used with numpy arrays directly

    """
    tag = -tag  #reverting the - operation first
    fragment_idx = tag >> 48
    fof_tag = tag & ((1<<48) - 1)
    return fof_tag, fragment_idx


@numba.jit(nopython=True)
def _fix_fragment_property_constrev(snapnum, fof_tag, data_in, data_out):
    nhalos = len(snapnum)
    lastsnap = snapnum[0]
    lastval = data_in[0]
    for i in range(nhalos):
        _index = 0
        _tag = fof_tag[i]
        if _tag < 0:
            _tag = -_tag
            _index = _tag >> 48
            _tag = _tag & ((1<<48) - 1)
        if _index == 0 or snapnum[i] >= lastsnap:
            lastval = data_in[i]
        data_out[i] = lastval
        lastsnap = snapnum[i]


@numba.jit(nopython=True)
def _fix_fragment_property_linear(snapnum, fof_tag, data_in, data_out, mask_neg):
    nhalos = len(snapnum)
    lastsnap = snapnum[0]
    start_index = 0
    for i in range(nhalos):
        _index = 0
        _tag = fof_tag[i]
        if _tag < 0:
            _tag = -_tag
            _index = _tag >> 48
            _tag = _tag & ((1<<48) - 1)

        if snapnum[i] >= lastsnap:
            # we are in a new branch, discard current fragment (should not be 
            # set anyway!)
            data_out[i] = data_in[i]
            start_index = i
        elif _index == 0:
            if start_index < i-1:
                # We just ended a fragment, need to go back and linearly interpolate!
                nsteps = i-start_index
                valstart = data_in[start_index]
                dval = (data_in[i] - valstart)/nsteps
                dval_mask = min(data_in[i], valstart)
                masked = mask_neg and dval_mask < 0
                for j in range(1, nsteps):
                    data_out[start_index + j] = (valstart + j*dval * (not masked)) + (dval_mask * masked)
            data_out[i] = data_in[i]
            start_index = i
        else:
            # We are in a non-main fragment, let's wait until we're a "real" object 
            pass


def fix_fragment_properties(
    forest: Mapping[str, np.ndarray], 
    keys: List[str],
    *,
    inplace: bool = True,
    suffix: str = '_fragfix',
    interpolation: str = 'constant_reverse',
    mask_negative: bool = False
    ) -> None:
    """Correct properties of minor fragments

    The SOD data stored in the merger tree forest files will be wrong for
    fragmented FoF groups, if the halo is not the major fragment of this group.
    This function attempts to fix those values by either linearly interpolate
    or setting the SOD properties to a constant value during the times when
    a halo is a minor fragment.

    Parameters
    ----------
    forest
        the full merger tree forest data

    keys
        list of the columns to which the correction should be applied

    inplace
        if True, the column will be updated in-place, otherwise, a new array
        will be created with the original name and suffix appended

    suffix
        if not in-place, the new array will be named ``old_column + suffix``

    interpolation
        the type of correction to apply. Currently supported are: 

        - ``"constant_reverse"``: set the properties to the value the halo has
          when becoming a non-minor-fragment or independent ("reverse" in time)
        - ``"linear"``: linearly interpolate (in snapshot-space) the property
          values from before and after being a minor fragment

    mask_negative
        if True, will not attempt to do a linear interpolation if either the 
        starting or ending value is negative, e.g. for an invalid concentration
        parameter. Instead, the property will be set to the negative value
        during the minor-fragment phase.

    Examples
    --------
    >>> haccytrees.mergertrees.fix_fragment_properties(forest, 
    ...     ['sod_halo_mass', 'sod_halo_radius', 'sod_halo_cdelta'],
    ...     inplace=False, 
    ...     suffix='_fragfix_const', 
    ...     interpolation='constant_reverse')
    >>> haccytrees.mergertrees.fix_fragment_properties(forest, 
    ...     ['sod_halo_mass', 'sod_halo_radius', 'sod_halo_cdelta'],
    ...     inplace=False, 
    ...     suffix='_fragfix_lin', 
    ...     interpolation='linear',
    ...     mask_negative=True)

    """
    for k in keys:
        data = forest[k]
        if inplace:
            data_out = data
        else:
            data_out = np.empty_like(data)
            forest[f'{k}{suffix}'] = data_out
        
        if interpolation == 'constant_reverse':
            _fix_fragment_property_constrev(forest['snapnum'], forest['fof_halo_tag'], data, data_out)
        elif interpolation == 'linear':
            _fix_fragment_property_linear(forest['snapnum'], forest['fof_halo_tag'], data, data_out, mask_negative)
        else:
            raise NotImplementedError(f"unknown interpolation: {interpolation}")