import sys
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def replace_low_confidence_locations(locations, confidences, confidence_threshold=0.5):
    locations_initial_shape = locations.shape
    # Squeeze out the last axis since we don't need it for computation
    n_frames, n_locations, n_coords, _ = locations.shape
    # Reshape locations and confidences for easier manipulation
    locations = locations.squeeze(-1)  # shape: (147545, 6, 2)
    confidences = confidences.squeeze(-1)  # shape: (147545, 6)
    # Create a mask for low-confidence locations
    low_confidence_mask = confidences < confidence_threshold  # shape: (147545, 6)
    # Loop over each point (i.e., keypoint) to handle missing values
    for point_idx in range(n_locations):
        for coord_idx in range(n_coords):  # For x and y coordinates
            # Extract the trajectory for this point and coordinate across frames
            trajectory = locations[:, point_idx, coord_idx]
            # Mask low-confidence points by setting them to NaN
            trajectory[low_confidence_mask[:, point_idx]] = np.nan
            # Find the indices of the valid (non-NaN) frames
            valid_idx = np.where(~np.isnan(trajectory))[0]
            valid_trajectory = trajectory[valid_idx]
            # Perform interpolation for missing points (at least two valid points required)
            if len(valid_idx) > 1:
                interp_func = interp1d(
                    valid_idx,
                    valid_trajectory,
                    kind="linear",
                    fill_value="extrapolate",
                    bounds_error=False,
                )
                # Replace the NaN with interpolated points
                locations[:, point_idx, coord_idx] = interp_func(np.arange(n_frames))
    locations = locations.reshape(locations_initial_shape)
    return locations


def main(argv):
    if len(argv) < 2:
        print(f"Usage: python {argv[0]} <h5 filename>")
        exit(0)
    filename = argv[1]
    output_filename = filename.replace(".h5", ".no_low_conf.h5")
    with h5py.File(filename, "r") as src_file:
        locations = src_file["tracks"][:].T
        confidences = src_file["point_scores"][:].T
        print("Interpolating low-confidence points...")
        new_locations = replace_low_confidence_locations(locations, confidences)
        with h5py.File(output_filename, "w") as dest_file:
            for name, dataset in src_file.items():
                if name == "tracks":
                    dest_file.create_dataset(name, data=new_locations.T)
                else:
                    dest_file.create_dataset(name, data=dataset[...])
    print(f"Saved {output_filename}.")


if __name__ == "__main__":
    main(sys.argv)
