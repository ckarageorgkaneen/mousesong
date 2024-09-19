"""
    undistort_video.py
    Undistort video using ffmpeg and opencv
    Mike Zheng
    220713
    https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    https://github.com/kkroening/ffmpeg-python/blob/master/examples/tensorflow_stream.py
    https://github.com/kkroening/ffmpeg-python/issues/284

    example:
    python draw_box_frame.py <video_filepath> <intrinsics_filepath> <coords_filepath> <output_filepath>
    python draw_box_frame.py Z:/usv_calls/03_div_cage_group01/16_20221206_um002_ch007_mf_noscent/01hr/original_video/2022-12-06-102445_cam0.mp4 Z:/usv_calls/03_div_cage_group01/16_20221206_um002_ch007_mf_noscent/01hr/process_video/cam_params/intrinsics_21415940.mat Z:/usv_calls/03_div_cage_group01/16_20221206_um002_ch007_mf_noscent/01hr/process_video/cam_params/cage_coords_21415940.yml Z:/usv_calls/03_div_cage_group01/16_20221206_um002_ch007_mf_noscent/01hr/process_video/qc_cam0.png

"""

import argparse
import ffmpeg
import os
import sys
import subprocess
import numpy as np
import cv2
import scipy.io
import yaml

def get_video_size(filename):
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])
    return width, height

def start_ffmpeg_process1(in_filename):
    args = (
        ffmpeg
        .input(in_filename)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)

def start_ffmpeg_process2(out_filename, width, height):
    args = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='rgb24', framerate=50, s='{}x{}'.format(width, height))
        .output(out_filename, **{'c:v':'h264_nvenc', 'profile:v':'high', 'preset':'slow'})
        .overwrite_output()
        .compile()
    )
    return subprocess.Popen(args, stdin=subprocess.PIPE)

def read_frame(process1, width, height):

    frame_size = width * height * 3
    in_bytes = process1.stdout.read(frame_size)
    if len(in_bytes) == 0:
        frame = None
    else:
        assert len(in_bytes) == frame_size
        frame = (
            np
            .frombuffer(in_bytes, np.uint8)
            .reshape([height, width, 3])
        )
    return frame

def write_frame(process2, frame):
    process2.stdin.write(
        frame
        .astype(np.uint8)
        .tobytes()
    )

def run(in_filename, out_filename, cam_params, newcameramtx):
    width, height = get_video_size(in_filename)
    process1 = start_ffmpeg_process1(in_filename)
    process2 = start_ffmpeg_process2(out_filename, width, height)
    while True:
        in_frame = read_frame(process1, width, height)
        if in_frame is None:
            break

        # undistort image
        out_frame = cv2.undistort(in_frame, cam_params['cameraMatrix'], cam_params['distCoeffs'], None, newcameramtx)
        # rotate image 180 degrees
        out_frame = cv2.rotate(out_frame, cv2.ROTATE_180)

        write_frame(process2, out_frame)

    process1.wait()
    process2.stdin.close()
    process2.wait()

def run_drawcage(in_filename, out_filename, cam_params, newcameramtx, coords):
    width, height = get_video_size(in_filename)
    process1 = start_ffmpeg_process1(in_filename)
    process2 = start_ffmpeg_process2(out_filename, width, height)
    while True:
        in_frame = read_frame(process1, width, height)
        if in_frame is None:
            break

        # undistort image
        out_frame = cv2.undistort(in_frame, cam_params['cameraMatrix'], cam_params['distCoeffs'], None, newcameramtx)
        # rotate image 180 degrees
        out_frame = cv2.rotate(out_frame, cv2.ROTATE_180)

        x = coords['image_cage']['x']
        y = coords['image_cage']['y']
        w = coords['image_cage']['w']
        h = coords['image_cage']['h']
        cv2.rectangle(out_frame, (x, y), (x + w, y + h), (0,255,0), 2)

        write_frame(process2, out_frame)

    process1.wait()
    process2.stdin.close()
    process2.wait()

def main(argv):

    # parse arguments
    parser = argparse.ArgumentParser(description='undistort and draw box coords on first frame of video')
    parser.add_argument('video_filepath', type=str, nargs=1)
    parser.add_argument('intrinsics_filepath', type=str, nargs=1)
    parser.add_argument('output_filepath', type=str, nargs=1)

    args = parser.parse_args()
    video_filepath = args.video_filepath[0]
    intrinsics_filepath = args.intrinsics_filepath[0]
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

    run(video_filepath, output_filepath, cam_params, newcameramtx)


if __name__ == '__main__':
    main(sys.argv)