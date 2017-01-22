'''
  Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
'''

import cv2
import argparse
import numpy as np
import glob
import pickle
from calibration import calculate_mtx_dist


def main():
  parser = argparse.ArgumentParser(description="Compute calibration matrix and distortion coefficients")
  parser.add_argument('--images_pattern', type=str, default='camera_cal/calibration*.jpg', help='glob pattern of calibration images')
  parser.add_argument('--save_file', type=str, default='dist_pickle.p', help='save file (for mtx and dist params)')
  parser.add_argument('--verbose', default=False, action='store_true', help='verbose flag')
  parser.add_argument('--nx', type=int, default=9, help='nx param')
  parser.add_argument('--ny', type=int, default=6, help='ny param')

  args = parser.parse_args()

  images_pattern = args.images_pattern
  mtx_dist_save_file = args.save_file
  verbose = args.verbose
  nx = args.nx
  ny = args.ny

  # Make a list of calibration images
  images = glob.glob(images_pattern)

  print("Calculating distortion matrix('mtx') and coefficients('dist') for files '%s'" % images_pattern)

  # Calculate calibration matrix
  mtx, dist = calculate_mtx_dist(images, nx=nx, ny=ny, verbose=verbose)


  # Save Distortion matrix and coefficient
  if verbose:
    print("Saving distortion matrix('mtx') and coefficients('dist') to the file '%s'" % mtx_dist_save_file)
  with open(mtx_dist_save_file, 'wb') as f:
      saved_obj = {"mtx": mtx, "dist" : dist}
      pickle.dump(saved_obj, f)

if __name__ == '__main__':
  main()
