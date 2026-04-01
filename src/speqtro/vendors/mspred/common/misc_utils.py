"""misc_utils.py -- vendored from ms-pred, imports fixed for speqtro.

Only inference-relevant utilities are included. Training-specific code
(ConsoleLogger, pytorch_lightning logger, setup_logger) is omitted.
"""
import sys
import copy
import logging
from pathlib import Path, PosixPath
import json
from itertools import groupby, islice
from typing import Tuple, List
import pandas as pd
import numpy as np

from . import chem_utils

# Try to import h5py -- it's needed for HDF5Dataset but may not always be
# available in minimal installs.
try:
    import h5py
except ImportError:
    h5py = None

NIST_COLLISION_ENERGY_MEAN = 40.260853377886264
NIST_COLLISION_ENERGY_STD = 31.604227557486197


def get_data_dir(dataset_name: str) -> Path:
    return Path("data/spec_datasets") / dataset_name


class HDF5Dataset:
    """
    A dataset as a HDF5 file
    """
    def __init__(self, path, mode="r"):
        if h5py is None:
            raise ImportError("h5py is required for HDF5Dataset")
        self.path = path
        self.h5_obj = h5py.File(path, mode=mode)
        self.attrs = self.h5_obj.attrs

    def __getitem__(self, idx):
        return self.h5_obj[idx]

    def __setitem__(self, key, value):
        self.h5_obj[key] = value

    def __contains__(self, idx):
        return idx in self.h5_obj

    def get_all_names(self):
        return self.h5_obj.keys()

    def read_str(self, name, encoding='utf-8') -> str:
        if '/' in name:  # has group
            groupname, name = name.rsplit('/', 1)
            grp = self.h5_obj[groupname]
        else:
            grp = self.h5_obj
        str_obj = grp[name][0]
        if type(str_obj) is not bytes:
            raise TypeError(f'Wrong type of {name}')
        return str_obj.decode(encoding)

    def write_str(self, name, data):
        if '/' in name:  # has group
            groupname, name = name.rsplit('/', 1)
            grp = self.h5_obj.require_group(groupname)
        else:
            grp = self.h5_obj
        dt = h5py.special_dtype(vlen=str)
        ds = grp.create_dataset(name, (1,), dtype=dt, compression="gzip")
        ds[0] = data

    def write_dict(self, dict):
        """dict entries: {filename: data}"""
        for filename, data in dict.items():
            self.write_str(filename, data)

    def write_list_of_tuples(self, list_of_tuples):
        """each tuple is (filename, data)"""
        for tup in list_of_tuples:
            if tup is None:
                continue
            self.write_str(tup[0], tup[1])

    def read_data(self, name) -> np.ndarray:
        """read a numpy array object"""
        return self.h5_obj[name][:]

    def write_data(self, name, data):
        """write a numpy array object"""
        self.h5_obj.create_dataset(name, data=data)

    def read_attr(self, name) -> dict:
        """read attribute of name as a dict"""
        return {k: v for k, v in self.h5_obj[name].attrs.items()}

    def update_attr(self, name, inp_dict):
        """write inp_dict to name's attribute"""
        cur_obj = self.h5_obj[name].attrs
        for k, v in inp_dict.items():
            cur_obj[k] = v

    def close(self):
        self.h5_obj.close()

    def flush(self):
        self.h5_obj.flush()


# -- Spectra parsing ----------------------------------------------------------

def parse_spectra(spectra_file) -> Tuple[dict, List[Tuple[str, np.ndarray]]]:
    """parse_spectra.

    Parses spectra in the SIRIUS format and returns

    Args:
        spectra_file (str or list): Name of spectra file to parse or lines of parsed spectra
    Return:
        Tuple[dict, List[Tuple[str, np.ndarray]]]: metadata and list of spectra
            tuples containing name and array
    """
    if type(spectra_file) is str or type(spectra_file) is PosixPath:
        lines = [i.strip() for i in open(spectra_file, "r").readlines()]
    elif type(spectra_file) is list:
        lines = [i.strip() for i in spectra_file]
    else:
        raise ValueError(f'type of variable spectra_file not understood, got {type(spectra_file)}')

    group_num = 0
    metadata = {}
    spectras = []
    my_iterator = groupby(
        lines, lambda line: line.startswith(">") or line.startswith("#")
    )

    for index, (start_line, lines) in enumerate(my_iterator):
        group_lines = list(lines)
        subject_lines = list(next(my_iterator)[1])
        # Get spectra
        if group_num > 0:
            spectra_header = group_lines[0].split(">")[1]
            peak_data = [
                [float(x) for x in peak.split()[:2]]
                for peak in subject_lines
                if peak.strip()
            ]
            # Check if spectra is empty
            if len(peak_data):
                peak_data = np.vstack(peak_data)
                # Add new tuple
                spectras.append((spectra_header, peak_data))
        # Get meta data
        else:
            entries = {}
            for i in group_lines:
                if " " not in i:
                    continue
                elif i.startswith("#INSTRUMENT TYPE"):
                    key = "#INSTRUMENT TYPE"
                    val = i.split(key)[1].strip()
                    entries[key[1:]] = val
                else:
                    start, end = i.split(" ", 1)
                    start = start[1:]
                    while start in entries:
                        start = f"{start}'"
                    entries[start] = end

            metadata.update(entries)
        group_num += 1

    if type(spectra_file) is str:
        metadata["_FILE_PATH"] = spectra_file
        metadata["_FILE"] = Path(spectra_file).stem
    return metadata, spectras


def max_inten_spec(spec, max_num_inten: int = 60, inten_thresh: float = 0):
    """max_inten_spec.

    Args:
        spec: 2D spectra array
        max_num_inten: Max number of peaks
        inten_thresh: Min intensity to allow in returned peak

    Return:
        Spec filtered down
    """
    spec_masses, spec_intens = spec[:, 0], spec[:, 1]

    # Sort by intensity and select top subpeaks
    new_sort_order = np.argsort(spec_intens)[::-1]
    if max_num_inten is not None:
        new_sort_order = new_sort_order[:max_num_inten]

    spec_masses = spec_masses[new_sort_order]
    spec_intens = spec_intens[new_sort_order]

    spec_mask = spec_intens > inten_thresh
    spec_masses = spec_masses[spec_mask]
    spec_intens = spec_intens[spec_mask]
    spec = np.vstack([spec_masses, spec_intens]).transpose(1, 0)
    return spec


def norm_spectrum(binned_spec: np.ndarray) -> np.ndarray:
    """norm_spectrum.

    Normalizes each spectral channel to have norm 1

    Args:
        binned_spec (np.ndarray) : Vector of spectras

    Return:
        np.ndarray where each channel has max(1)
    """
    spec_maxes = binned_spec.max(1)
    non_zero_max = spec_maxes > 0
    spec_maxes = spec_maxes[non_zero_max]
    binned_spec[non_zero_max] = binned_spec[non_zero_max] / spec_maxes.reshape(-1, 1)
    binned_spec = np.sqrt(binned_spec)
    return binned_spec


def process_spec_file(meta, tuples, precision=4, merge_specs=True, exclude_parent=False):
    """process_spec_file."""

    parent_mass = meta.get("parentmass", None)
    if parent_mass is None:
        print(f"missing parentmass for spec")
        parent_mass = 1000000

    parent_mass = float(parent_mass)

    # First norm spectra
    fused_tuples = {ce: x for ce, x in tuples if x.size > 0}

    if len(fused_tuples) == 0:
        return

    if merge_specs:
        mz_to_inten_pair = {}
        new_tuples = []
        for i in fused_tuples.values():
            for tup in i:
                mz, inten = tup
                mz_ind = np.round(mz, precision)
                cur_pair = mz_to_inten_pair.get(mz_ind)
                if cur_pair is None:
                    mz_to_inten_pair[mz_ind] = tup
                    new_tuples.append(tup)
                elif inten > cur_pair[1]:
                    cur_pair[1] = inten  # max merging
                else:
                    pass

        merged_spec = np.vstack(new_tuples)
        if exclude_parent:
            merged_spec = merged_spec[merged_spec[:, 0] <= (parent_mass - 1)]
        else:
            merged_spec = merged_spec[merged_spec[:, 0] <= (parent_mass + 1)]
        merged_spec = merged_spec[merged_spec[:, 1] > 0]
        if len(merged_spec) == 0:
            return
        merged_spec[:, 1] = merged_spec[:, 1] / merged_spec[:, 1].max()

        # Sqrt intensities here
        merged_spec[:, 1] = np.sqrt(merged_spec[:, 1])
        return merged_spec
    else:
        new_specs = {}
        for k, v in fused_tuples.items():
            new_spec = np.vstack(v)
            new_spec = new_spec[new_spec[:, 0] <= (parent_mass + 1)]
            new_spec = new_spec[new_spec[:, 1] > 0]
            if len(new_spec) == 0:
                continue
            new_spec[:, 1] = new_spec[:, 1] / new_spec[:, 1].max()
            new_spec[:, 1] = np.sqrt(new_spec[:, 1])
            new_specs[k] = new_spec
        return new_specs


def bin_spectra(
    spectras: List[np.ndarray],
    num_bins: int = 15000,
    upper_limit: int = 1500,
    pool_fn: str = "max",
) -> np.ndarray:
    """bin_spectra."""
    if pool_fn == "add":
        pool = lambda x, y: x + y
    elif pool_fn == "max":
        pool = lambda x, y: max(x, y)
    else:
        raise NotImplementedError()

    bins = np.linspace(0, upper_limit, num=num_bins)
    binned_spec = np.zeros((len(spectras), len(bins)))
    for spec_index, spec in enumerate(spectras):
        digitized_mz = np.digitize(spec[:, 0], bins=bins)
        in_range = digitized_mz < len(bins)
        digitized_mz, spec = digitized_mz[in_range], spec[in_range, :]
        for bin_index, spec_val in zip(digitized_mz, spec[:, 1]):
            cur_val = binned_spec[spec_index, bin_index]
            binned_spec[spec_index, bin_index] = pool(spec_val, cur_val)

    return binned_spec


def merge_specs(specs_list, precision=4, merge_method='sum'):
    mz_to_inten_pair = {}
    new_tuples = []
    for spec in specs_list.values():
        for tup in spec:
            mz, inten = tup
            mz_ind = np.round(mz, precision)
            cur_pair = mz_to_inten_pair.get(mz_ind)
            if cur_pair is None:
                mz_to_inten_pair[mz_ind] = tup
                new_tuples.append(tup)
            else:
                if merge_method == 'sum':
                    cur_pair[1] += inten
                elif merge_method == 'max':
                    cur_pair[1] = max(cur_pair[1], inten)
                else:
                    raise ValueError(f'Unknown merge_method {merge_method}')

    merged_spec = np.vstack(new_tuples)
    merged_spec = merged_spec[merged_spec[:, 1] > 0]
    if len(merged_spec) == 0:
        return
    merged_spec[:, 1] = merged_spec[:, 1] / merged_spec[:, 1].max()

    return {'nan': merged_spec}


def merge_intens(spec_dict):
    merged_intens = np.zeros_like(next(iter(spec_dict.values())))
    for spec in spec_dict.values():
        merged_intens += spec
    merged_intens = merged_intens / merged_intens.max()
    return {'nan': merged_intens}


def batches(it, chunk_size: int):
    """Consume an iterable in batches of size chunk_size"""
    it = iter(it)
    return iter(lambda: list(islice(it, chunk_size)), [])


def np_stack_padding(it, axis=0):
    def resize(row, size):
        new = np.array(row)
        new.resize(size)
        return new

    max_shape = [max(i) for i in zip(*[j.shape for j in it])]
    mat = np.stack([resize(row, max_shape) for row in it], axis=axis)
    return mat


def get_collision_energy(filename):
    import re as _re
    colli_eng = _re.findall('collision +([0-9]+\\.?[0-9]*|nan).*', filename)
    if len(colli_eng) > 1:
        raise ValueError(f'Multiple collision energies found in {filename}')
    if len(colli_eng) == 1:
        colli_eng = colli_eng[0].split()[-1]
    else:
        colli_eng = 'nan'
    return colli_eng


import hashlib

def str_to_hash(inp_str, digest_size=16):
    return hashlib.blake2b(inp_str.encode("ascii"), digest_size=digest_size).hexdigest()


def rm_collision_str(key: str) -> str:
    """remove `_collision VALUE` from the string"""
    keys = key.split('_collision')
    if len(keys) == 2:
        return keys[0]
    elif len(keys) == 1:
        return key
    else:
        raise ValueError(f'Unrecognized key: {key}')


def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False
