"""
make_vid_sleap.py
Overlay sleap pose predictions on undistorted videos with custom style
similar to draw_box_video.py in divided_cage
Mike Zheng
220727

https://github.com/kkroening/ffmpeg-python/blob/master/examples/tensorflow_stream.py
https://github.com/kkroening/ffmpeg-python/issues/284

example:
python make_vid_sleap.py <video_filepath> <interpolate_filepath> <color> <output_filepath>
python make_vid_sleap.py Z:/usv_calls/03_div_cage_group01/16_20221206_um002_ch007_mf_noscent/01hr/process_video/2022-12-06-102445_cam0_undistort.mp4 Z:/usv_calls/03_div_cage_group01/16_20221206_um002_ch007_mf_noscent/01hr/process_video/2022-12-06-102445_cam0_undistort.predictions.analysis.interpolate.mat green Z:/usv_calls/03_div_cage_group01/16_20221206_um002_ch007_mf_noscent/01hr/original_video/2022-12-06-102445_cam0_undistort.pose.mp4


"""

import argparse
import ffmpeg
import sys
import subprocess
import numpy as np
import cv2
import scipy.io

color_dict = {
    "green": (0, 255, 0),
    "magenta": (255, 0, 255),
}


def get_video_size(filename):
    probe = ffmpeg.probe(filename)
    video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
    width = int(video_info["width"])
    height = int(video_info["height"])
    return width, height


def start_ffmpeg_process1(in_filename):
    args = (
        ffmpeg.input(in_filename)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .compile()
    )
    return subprocess.Popen(args, stdout=subprocess.PIPE)


def start_ffmpeg_process2(out_filename, width, height):
    args = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            framerate=50,
            s="{}x{}".format(width, height),
        )
        .output(
            out_filename, **{"c:v": "h264_nvenc", "profile:v": "high", "preset": "slow"}
        )
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
        frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
    return frame


def write_frame(process2, frame):
    process2.stdin.write(frame.astype(np.uint8).tobytes())


def run_plot_pose(in_filename, out_filename, locations, skeleton, color):
    i_frame = 0

    width, height = get_video_size(in_filename)
    process1 = start_ffmpeg_process1(in_filename)
    process2 = start_ffmpeg_process2(out_filename, width, height)
    while True:
        frame = read_frame(process1, width, height)
        if frame is None:
            break

        plot_pose_image_overlay(frame, locations[i_frame, :], skeleton, color)

        write_frame(process2, frame)

        i_frame += 1

    process1.wait()
    process2.stdin.close()
    process2.wait()


def plot_pose_image_overlay(image, location_frame, skeleton, color):
    xs = location_frame[:, 0].astype(int)
    ys = location_frame[:, 1].astype(int)

    # plot joints
    for i_joint in range(skeleton.shape[0]):
        index1, index2 = skeleton[i_joint, 0], skeleton[i_joint, 1]
        cv2.line(
            image,
            [xs[index1], ys[index1]],
            [xs[index2], ys[index2]],
            color=color,
            thickness=5,
        )

    # plot points
    for j in range(location_frame.shape[0]):
        cv2.circle(image, [xs[j], ys[j]], 5, (255, 255, 255), -1)


def main(argv):
    # parse arguments
    parser = argparse.ArgumentParser(description="make sleap video")
    parser.add_argument("video_filepath", type=str, nargs=1)
    parser.add_argument("interpolate_filepath", type=str, nargs=1)
    parser.add_argument("color", type=str, nargs=1)
    parser.add_argument("output_filepath", type=str, nargs=1)

    args = parser.parse_args()
    video_filepath = args.video_filepath[0]
    interpolate_filepath = args.interpolate_filepath[0]
    color = args.color[0]
    output_filepath = args.output_filepath[0]

    # read interpolate location
    locations = scipy.io.loadmat(interpolate_filepath, squeeze_me=True)["locations"]

    # define skeleton
    skeleton = np.array([[2, 0], [2, 1], [0, 1], [2, 3], [3, 4], [4, 5]])

    # vidcap = cv2.VideoCapture(video_filepath)
    # success,image = vidcap.read()

    # plot_pose_image_overlay(image, locations[0,:], skeleton, color_dict[color])

    # cv2.imshow('test', image)
    # cv2.imwrite('test.png', image)

    run_plot_pose(
        video_filepath, output_filepath, locations, skeleton, color_dict[color]
    )


if __name__ == "__main__":
    main(sys.argv)
