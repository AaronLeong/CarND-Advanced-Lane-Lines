'''
  Find Lane Lines Pipeline
'''

import cv2
import argparse
import numpy as np
import glob
import pickle
from moviepy.editor import VideoFileClip
from calibration import calculate_mtx_dist


def process_func(img):
  img_copy = np.copy(img)
  return img_copy


def main():
  parser = argparse.ArgumentParser(description="Find Lane Lines on a video")
  parser.add_argument('--video', type=str, default='project_video.mp4', help='project video')
  parser.add_argument('--output_video', default='output.mp4', type=str, help='output video')
  parser.add_argument('--mtx_dist', type=str, default='dist_pickle.p', help='saved file (for mtx and dist params)')
  parser.add_argument('--verbose', default=False, action='store_true', help='verbosity flag')

  args = parser.parse_args()

  video_file = args.video
  output_video_file = args.output_video
  mtx_dist_file = args.mtx_dist
  verbose = args.verbose

  print("Video file: {}".format(video_file))
  print("Output video file: {}".format(output_video_file))
  print("Mtx/Dist file: {}".format(mtx_dist_file))
  print("Verbose: {}".format(verbose))

  print("Find lane lines ...")

  # Load Saved Camera Matrix and Distortion Coefficients
  dist_pickle = pickle.load(open(mtx_dist_file, "rb" ))
  mtx = dist_pickle["mtx"]
  dist = dist_pickle["dist"]

  if verbose:
    print('mtx=',mtx)
    print('dist=',dist)


  clip = VideoFileClip(video_file)
  clip = clip.fl_image(process_func)
  clip.write_videofile(output_video_file, audio=False)

if __name__ == '__main__':
  main()
