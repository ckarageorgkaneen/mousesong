"""
    draw_box_frame.py
    Read in a video, camera intrinsics, and cage coords
    Output the first frame, undistorted and draw box
    Quality control for the intrinsics and the find_cage_coords across sessions

    example:
    python draw_box_frame.py <video_filepath> <intrinsics_filepath> <coords_filepath> <output_filepath>
    python draw_box_frame.py Z:/usv_calls/03_div_cage_group01/16_20221206_um002_ch007_mf_noscent/01hr/original_video/2022-12-06-102445_cam0.mp4 Z:/usv_calls/03_div_cage_group01/16_20221206_um002_ch007_mf_noscent/01hr/process_video/cam_params/intrinsics_21415940.mat Z:/usv_calls/03_div_cage_group01/16_20221206_um002_ch007_mf_noscent/01hr/process_video/cam_params/cage_coords_21415940.yml Z:/usv_calls/03_div_cage_group01/16_20221206_um002_ch007_mf_noscent/01hr/process_video/qc_cam0.png

"""

import argparse
import cv2
import ffmpeg
import os
import sys
import yaml
import numpy as np
import scipy.io

def get_video_size(filename):
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height

def main(argv):

    # parse arguments
    parser = argparse.ArgumentParser(description='undistort and draw box coords on first frame of video')
    parser.add_argument('video_filepath', type=str, nargs=1)
    parser.add_argument('intrinsics_filepath', type=str, nargs=1)
    parser.add_argument('coords_filepath', type=str, nargs=1)
    parser.add_argument('output_filepath', type=str, nargs=1)

    args = parser.parse_args()
    video_filepath = args.video_filepath[0]
    intrinsics_filepath = args.intrinsics_filepath[0]
    coords_filepath = args.coords_filepath[0]
    output_filepath = args.output_filepath[0]

    # read camera params
    cam_params_mat = scipy.io.loadmat(intrinsics_filepath)
    cam_params = {}
    # for intrinsic matrix, convert from matlab format to opencv format
    # transpose
    # make principal point (0,0) instead of (1,1)
    cam_params['cameraMatrix'] = cam_params_mat['K'].T
    cam_params['cameraMatrix'][0,2] -= 1
    cam_params['cameraMatrix'][1,2] -= 1
    # concatenate RDistort and TDistort to make the distCoeffs
    cam_params['distCoeffs'] = np.hstack([cam_params_mat['RDistort'].flatten(), cam_params_mat['TDistort'].flatten()])

    # refine camera matrix
    width, height = get_video_size(video_filepath)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_params['cameraMatrix'], cam_params['distCoeffs'], (width,height), 1, (width,height))

    # read cage coords
    with open(coords_filepath, "r") as f:
        coords = yaml.safe_load(f)

    # read first frame of the video
    cap = cv2.VideoCapture(video_filepath)
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # undistort
    out_frame = cv2.undistort(frame, cam_params['cameraMatrix'], cam_params['distCoeffs'], None, newcameramtx)
    out_frame = cv2.rotate(out_frame, cv2.ROTATE_180)
    x = coords['image_cage']['x']
    y = coords['image_cage']['y']
    w = coords['image_cage']['w']
    h = coords['image_cage']['h']
    cv2.rectangle(out_frame, (x, y), (x + w, y + h), (0,255,0), 2)

    cv2.imshow('out_frame', out_frame)
    cv2.imwrite(output_filepath, out_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main(sys.argv)