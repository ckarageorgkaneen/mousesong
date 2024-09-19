# sleap_interpolate_missing_data.py
#
# read the export data from sleap
# interpolate missing data
#
# https://sleap.ai/notebooks/Analysis_examples.html
#
# Mike Zheng
# 220122

import sys

import h5py
import numpy as np

from scipy.interpolate import interp1d
from scipy.io import savemat


def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y


def main(argv):
    if len(argv) < 2:
        print(f"Usage: python {argv[0]} <h5 filename>")
        exit(0)

    filename = argv[1]

    output_filename = filename[:-3] + ".interpolate.mat"

    ## Loading the data

    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]

    print("===filename===")
    print(filename)
    print()

    print("===HDF5 datasets===")
    print(dset_names)
    print()

    print("===locations data shape===")
    print(locations.shape)
    print()

    print("===nodes===")
    for i, name in enumerate(node_names):
        print(f"{i}: {name}")
    print()

    print("===tracking summary===")

    frame_count, node_count, _, instance_count = locations.shape

    print("frame count:", frame_count)
    print("node count:", node_count)
    print("instance count:", instance_count)

    ## Fill missing values

    new_locations = fill_missing(locations)

    ## save new locations to mat file
    savemat(output_filename, {"locations": new_locations})


if __name__ == "__main__":
    main(sys.argv)
