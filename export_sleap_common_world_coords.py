"""
export_sleap_common_coords_world.py
Export tracking of two animals in common coordinates based on calibration
Mike Zheng
220303

example:
python export_sleap_common_coords_world.py <l_interpolate_filepath> <l_coords_filepath> <r_interpolate_filepath> <r_coords_filepath> <output_filepath>

"""

import argparse
import sys
import numpy as np
import scipy.io
import yaml


def main(argv):
    # parse arguments
    parser = argparse.ArgumentParser(
        description="combine sleap videos and put them into common coords"
    )
    parser.add_argument("l_interpolate_filepath", type=str, nargs=1)
    parser.add_argument("l_coords_filepath", type=str, nargs=1)
    parser.add_argument("r_interpolate_filepath", type=str, nargs=1)
    parser.add_argument("r_coords_filepath", type=str, nargs=1)
    parser.add_argument("output_filepath", type=str, nargs=1)

    args = parser.parse_args()
    l_interpolate_filepath = args.l_interpolate_filepath[0]
    l_coords_filepath = args.l_coords_filepath[0]
    r_interpolate_filepath = args.r_interpolate_filepath[0]
    r_coords_filepath = args.r_coords_filepath[0]
    output_filepath = args.output_filepath[0]

    # image center
    img_center = (808, 620)
    # z ratio, used to correct z
    z_ratio = (12 - 1.5) / 12  # (cam_z-mouse_z)/cam_z

    # process left side
    l_locations_img = scipy.io.loadmat(l_interpolate_filepath, squeeze_me=True)[
        "locations"
    ]
    # correct z
    l_locations_img_zcorrect = np.zeros_like(l_locations_img)
    l_locations_img_zcorrect[:, :, 0] = (
        l_locations_img[:, :, 0] - img_center[0]
    ) * z_ratio + img_center[0]
    l_locations_img_zcorrect[:, :, 1] = (
        l_locations_img[:, :, 1] - img_center[1]
    ) * z_ratio + img_center[1]
    with open(l_coords_filepath, "r") as f:
        l_coords = yaml.safe_load(f)
    l_coords_img = l_coords["image_cage"]
    l_coords_world = l_coords["world_cage"]
    l_locations_world = np.zeros_like(l_locations_img_zcorrect)
    l_locations_world[:, :, 0] = (
        l_locations_img_zcorrect[:, :, 0] - l_coords_img["x"]
    ) * l_coords_world["w"] / l_coords_img["w"] + l_coords_world["x"]
    l_locations_world[:, :, 1] = (
        l_locations_img_zcorrect[:, :, 1] - l_coords_img["y"]
    ) * l_coords_world["h"] / l_coords_img["h"] + l_coords_world["y"]

    # process right side
    r_locations_img = scipy.io.loadmat(r_interpolate_filepath, squeeze_me=True)[
        "locations"
    ]
    # correct z
    r_locations_img_zcorrect = np.zeros_like(r_locations_img)
    r_locations_img_zcorrect[:, :, 0] = (
        r_locations_img[:, :, 0] - img_center[0]
    ) * z_ratio + img_center[0]
    r_locations_img_zcorrect[:, :, 1] = (
        r_locations_img[:, :, 1] - img_center[1]
    ) * z_ratio + img_center[1]
    with open(r_coords_filepath, "r") as f:
        r_coords = yaml.safe_load(f)
    r_coords_img = r_coords["image_cage"]
    r_coords_world = r_coords["world_cage"]
    r_locations_world = np.zeros_like(r_locations_img_zcorrect)
    r_locations_world[:, :, 0] = (
        r_locations_img_zcorrect[:, :, 0] - r_coords_img["x"]
    ) * r_coords_world["w"] / r_coords_img["w"] + r_coords_world["x"]
    r_locations_world[:, :, 1] = (
        r_locations_img_zcorrect[:, :, 1] - r_coords_img["y"]
    ) * r_coords_world["h"] / r_coords_img["h"] + r_coords_world["y"]

    scipy.io.savemat(
        output_filepath,
        {
            "l_locations_world": l_locations_world,
            "r_locations_world": r_locations_world,
        },
    )


if __name__ == "__main__":
    main(sys.argv)
