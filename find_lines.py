'''
  Find Lane Lines Pipeline
'''

import cv2
import argparse
import numpy as np
import glob
import pickle
from scipy import misc
from moviepy.editor import VideoFileClip
from calibration import calculate_mtx_dist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(0,255)):

  # 1) Convert to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  # 2) Take the derivative in x or y given orient = 'x' or 'y'
  if orient == 'y':
    sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
  else:
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
  # 3) Take the absolute value of the derivative or gradient
  sobel_abs = np.absolute(sobel)
  # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
  sobel_scaled = np.uint8(255*sobel_abs/np.max(sobel_abs))
  # 5) Create a mask of 1's where the scaled gradient magnitude
  # is > thresh_min and < thresh_max
  msk = np.zeros_like(sobel_scaled)
  msk[(sobel_scaled > thresh[0]) & (sobel_scaled < thresh[1]) ] = 1
  # print(msk)
  # 6) Return this mask as your binary_output image
  return msk

def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
  # 1) Convert to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  # 2) Take the gradient in x and y separately
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
  sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
  # 3) Calculate the magnitude
  sobel_magn = np.sqrt(sobelx * sobelx + sobely * sobely)
  # 5) Scale to 8-bit (0 - 255) and convert to type = np.uint8
  sobel_scaled = np.uint8(255*sobel_magn/np.max(sobel_magn))
  # 6) Create a binary mask where mag thresholds are met
  msk = np.zeros_like(sobel_scaled)
  msk[(sobel_scaled > mag_thresh[0]) & (sobel_scaled < mag_thresh[1]) ] = 1
  # 7) Return this mask as your binary_output image
  return msk

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
  # 1) Convert to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  # 2) Take the gradient in x and y separately
  sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
  sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
  # 3) Take the absolute value of the x and y gradients
  abs_sobelx = np.absolute(sobelx)
  abs_sobely = np.absolute(sobely)
  # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
  dir1 = np.arctan2(abs_sobely, abs_sobelx)
  # 5) Create a binary mask where direction thresholds are met
  msk = np.uint8(np.zeros_like(dir1))
  msk[(dir1 >= thresh[0]) & (dir1 <= thresh[1])] = 1
  # 6) Return this mask as your binary_output image
  return msk

# Thresholds the S-channel of HLS
def hls_select(img, thresh=(0, 255)):
  # 1) Convert to HLS color space
  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
  H = hls[:,:,0]
  L = hls[:,:,1]
  S = hls[:,:,2]
  # 2) Apply a threshold to the S channel
  bin = np.zeros_like(S)
  bin[(S > thresh[0]) & (S <= thresh[1])] = 1
  return bin

def apply_all_thresholds(img):
  ksize = 31
  thresh_sobel = (50, 150)
  thresh_mag = (50, 255)
  thresh_dir = (0.75, 1.15)

  # Gradient, Magnitude, Direction Thresholds
  gradx = sobel_threshold(img, orient='x', sobel_kernel=ksize, thresh=thresh_sobel)
  mag_bin = mag_threshold(img, sobel_kernel=ksize, mag_thresh=thresh_mag)
  dir_bin = dir_threshold(img, sobel_kernel=ksize, thresh=thresh_dir)

  # Combine Thresholds 1
  comb_bin = np.zeros_like(gradx)
  comb_bin[(gradx == 1) | ((dir_bin == 1) & (mag_bin == 1))] = 1

  # Color Threshold S-channel
  thresh_s = (170, 255)

  hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
  img_color = hls[:,:,2]

  color_bin = np.zeros_like(img_color)
  color_bin[(img_color > thresh_s[0]) & (img_color <= thresh_s[1])] = 1

  # Color Threshold R-channel
  thresh_r = (200, 255)
  r_img = img[:,:,0]

  r_bin = np.zeros_like(r_img)
  r_bin[(r_img > thresh_r[0]) & (r_img <= thresh_r[1])] = 1

  # Combined Gradient/Mag + Color S + Color R
  combined = np.zeros_like(comb_bin)
  combined[(comb_bin == 1) | (color_bin == 1) | (r_bin == 1)] = 1

  return combined



def compose_images(dst, src, nrows, ncols, num):
  assert 0 < num <= nrows * ncols

  if nrows > ncols:
      newH = int(dst.shape[0]/nrows)
      dim = (int(dst.shape[1] * newH/dst.shape[0]), newH)
  else:
      newW = int(dst.shape[1]/ncols)
      dim = (newW, int(dst.shape[0] * newW/dst.shape[1]))

  # Make it work for bin images too
  if len(src.shape) == 2:
      srcN = bin_to_rgb(src)
  else:
      srcN = np.copy(src)

  img = cv2.resize(srcN, dim, interpolation = cv2.INTER_AREA)
  nr = (num - 1) // ncols
  nc = (num - 1) % ncols
  dst[nr * img.shape[0]:(nr + 1) * img.shape[0], nc * img.shape[1]:(nc + 1) * img.shape[1]] = img
  return dst


def get_fig_image(fig):
  # http://www.itgo.me/a/1944619462852588132/matplotlib-save-plot-to-numpy-array
  # This is very BAD hack ....
  fig.savefig('output_images/tmp.png')
  # fig.canvas.draw()
  # data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  # data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  # plt.close(fig)
  img = cv2.imread('output_images/tmp.png')
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  # from scipy import misc
  # img = misc.imread('output_images/temp.png')
  return img

def bin_to_rgb(bin_image):
  return cv2.cvtColor(bin_image*255, cv2.COLOR_GRAY2RGB)

def fit_getx(fit, y):
  return fit[0]*y**2 + fit[1]*y + fit[2]

def curve_rad(fit, y):
  return ((1 + (2*fit[0]*y + fit[1])**2)**1.5) / np.absolute(2*fit[0])


def scale_fit(fit, yratio, xratio):
  return np.array([fit[0]*xratio/(yratio**2), fit[1]*xratio/yratio, fit[2]*xratio])


def find_peaks(bin_img):
  histogram = np.sum(bin_img[bin_img.shape[0]/2:,:], axis=0)

  hist_ind = np.argsort(histogram)
  hist_ind = hist_ind[::-1]
#         print('hist_ind =', hist_ind)
  hist_min_dist = 500
  # Search for 2 peaks
  peaks = []
  for ind in hist_ind:
      if len(peaks) == 0:
          peaks.append(ind)
          continue
      if len(peaks) == 1:
          if abs(ind-peaks[0]) > hist_min_dist:
              peaks.append(ind)
              break
  peaks = np.sort(peaks)
  return peaks, histogram

def save_output_img(img, fname):
  # Make it work for bin images too
  if len(img.shape) == 2:
      imgN = bin_to_rgb(img)
  else:
      imgN = img
  misc.imsave('output_images/output_%s.png' % fname, imgN)


class LineFinder(object):
  def __init__(self, mtx, dist, sampling=0.1):
    self.latest_fit_age = 1000 # arbitrary big number
    self.latest_left_fit = []
    self.latest_right_fit = []
    self.counter = 0
    self.mtx = mtx
    self.dist = dist
    self.M = None
    self.Minv = None
    self.sampling = sampling
    self.samples_orig = []
    self.samples_proc = []
    self.w = 0
    self.h = 0

  def setCurrentImage(self, image):
    self.current_image = image
    self.w, self.h = w, h = image.shape[1], image.shape[0]

    bottomW = w
    topW = 249 # 235 ((1180+100)*(180/1180))
    bottomH = h - 20
    topH = bottomH - 228 # h//2 + 100
    deltaW = 0 #

    self.region_vertices = np.array([[((w - bottomW) // 2 + deltaW, bottomH),
                                 ((w - topW) // 2 + deltaW, topH),
                                 ((w + topW) // 2 + deltaW, topH),
                                 ((w + bottomW) // 2 + deltaW, bottomH)]])


    offsetH = 10
    offsetW = 100
    self.dest_vertices = np.array([[(offsetW, h - offsetH),
                    (offsetW, offsetH),
                    (w - offsetW, offsetH),
                    (w - offsetW, h - offsetH)]])

    self.calculateTransformMatrices()

  def calculateTransformMatrices(self):
    if (self.M is None) or (self.Minv is None):
        self.M = cv2.getPerspectiveTransform(np.float32(self.region_vertices), np.float32(self.dest_vertices))
        self.Minv = cv2.getPerspectiveTransform(np.float32(self.dest_vertices), np.float32(self.region_vertices))

  def warpTransform(self, img):
    return cv2.warpPerspective(img, self.M, (self.w, self.h), flags=cv2.INTER_LINEAR)

  def invWarpTransform(self, img):
    return cv2.warpPerspective(img, self.Minv, (self.w, self.h), flags=cv2.INTER_LINEAR)

  def updateLatestFit(self, left_fit, right_fit):
    latest_weight = 0.9
    if self.latest_fit_age < 5:
        self.latest_left_fit = self.latest_left_fit * latest_weight + left_fit * (1 - latest_weight)
        self.latest_right_fit = self.latest_right_fit * latest_weight + right_fit * (1 - latest_weight)
    else:
        self.latest_left_fit = left_fit
        self.latest_right_fit = right_fit
    self.latest_fit_age = 0


  def calcCurvaturesAndCenter(self):
    # Calculate Curvature
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 20/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/750 # meters per pixel in x dimension

    # Scale fit to the real world
    left_fit_cr = scale_fit(self.latest_left_fit, ym_per_pix, xm_per_pix)
    right_fit_cr = scale_fit(self.latest_right_fit, ym_per_pix, xm_per_pix)

    y_eval =self.current_image.shape[0]-1
    self.left_curverad = curve_rad(left_fit_cr, y_eval * ym_per_pix)
    self.right_curverad = curve_rad(right_fit_cr, y_eval * ym_per_pix)
#         print('curves(m) =', left_curverad, right_curverad)

    # ============ Distance from center
    center_point = (fit_getx(self.latest_right_fit, y_eval) + fit_getx(self.latest_left_fit, y_eval))/2
    self.center_distance = (self.w/2 - center_point) * xm_per_pix
#         print('center_dist (m) =', center_distance)


  def drawLinesRegionOnRoad(self, img):
    # Generate x and y values for plotting
    fity = np.linspace(0, self.h-1, self.h )
    fit_leftx = fit_getx(self.latest_left_fit, fity)
    fit_rightx = fit_getx(self.latest_right_fit, fity)

    # Create an image to draw the lines on
    color_warp = np.zeros_like(img).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([fit_leftx, fity]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([fit_rightx, fity])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = self.invWarpTransform(color_warp)
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    return result


  def drawLines(self, img):
    # Generate x and y values for plotting
    fity = np.linspace(0, self.h-1, self.h )
    fit_leftx = fit_getx(self.latest_left_fit, fity)
    fit_rightx = fit_getx(self.latest_right_fit, fity)

    # Plot fitted lines
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(img)
    plt.plot(fit_leftx, fity, color='yellow', linewidth=4)
    plt.plot(fit_rightx, fity, color='yellow', linewidth=4)
    plt.title('Fit lines')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    data = get_fig_image(fig)
    return data



  def process_image(self, image):

    self.setCurrentImage(image)

    resImg = np.zeros_like(image)

    # Show: Original Image
    compose_images(resImg, image, 2, 2, 3)

    sampled = True if np.random.uniform() < self.sampling else False
    if sampled:
        self.samples_orig.append(image)

    # Frame size
    w, h = self.w, self.h

    # Undistort
    undist = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

    # Show: Undistorted image
    compose_images(resImg, undist, 4, 4, 1)
    if sampled:
      save_output_img(undist, '02%d_undist' % self.counter)


    # Thresholds: Gradient, Magnitude and Colors
    combined = apply_all_thresholds(undist)

    # Show: Thresholded Combined
    compose_images(resImg, combined, 4, 4, 2)
    if sampled:
      save_output_img(combined, '02%d_combined' % self.counter)


    # Show: Selected Region
    region_img = np.copy(image)
    cv2.polylines(region_img, self.region_vertices, True, (0, 0, 255), 5)
    cv2.polylines(region_img, self.dest_vertices, True, (0, 255, 0), 2)
    compose_images(resImg, region_img, 4, 4, 5)
    if sampled:
      save_output_img(region_img, '02%d_sel_region' % self.counter)



    # Transform to 'bird's view'
    combined_warped = self.warpTransform(combined)

    # Show: Warped Thresholded Combined
    compose_images(resImg, combined_warped, 4, 4, 3)
    if sampled:
      save_output_img(combined_warped, '02%d_comb_warped' % self.counter)


    # Show: Warped Original
    wimg = self.warpTransform(image)
    compose_images(resImg, wimg, 4, 4, 4)
    if sampled:
      save_output_img(wimg, '02%d_warped_orig' % self.counter)



    # ================ Search for 2 initial peaks
    peaks, histogram = find_peaks(combined_warped)

    # Show: Histogram
    fig = plt.figure(figsize=(10, 5))
    plt.plot(histogram)
    plt.plot(peaks, histogram[peaks,])
    data = get_fig_image(fig)
    compose_images(resImg, data, 4, 4, 6)
    if sampled:
      save_output_img(data, '02%d_histogram' % self.counter)


#         peaks = find_peaks_cwt(histogram, [10])
#         peaks = peakutils.indexes(np.int_(histogram), thres=0.05, min_dist=300)
#         peaks, peaks1 = peakdet(histogram, 0.2/max(histogram))


    numH = 10
    windW = 150
    windH = int(h/numH)


    # Look for left/right line points
    leftx = np.array([])
    lefty = np.array([])
    rightx = np.array([])
    righty = np.array([])

    # Prepare image for drawing left/right points
    combined_warped_img = bin_to_rgb(combined_warped)

    for wi in range(numH):

        wind = numH - wi
        botH = windH * wind
        topH = botH - windH

        # If we have a previous latest fit then use it to find peaks to start search from
        if self.latest_fit_age < 5:
            peaks[0] = fit_getx(self.latest_left_fit, botH-1)
            peaks[1] = fit_getx(self.latest_right_fit, botH-1)
            # Add points from the line to the point set at the beginning
            # acts like an anchor for a current fit and prevents line
            # from curving when there not enough data pixels near the car
            if wi == 0:
                addy = np.linspace(topH, botH-1, botH - topH )
                add_left_x = fit_getx(self.latest_left_fit, addy)
                add_right_x = fit_getx(self.latest_right_fit, addy)
                leftx = np.concatenate([leftx, add_left_x])
                lefty = np.concatenate([lefty, addy])
                rightx = np.concatenate([rightx, add_right_x])
                righty = np.concatenate([righty, addy])


        # Looking around from starting point to find optimal
        deltaW = windW//2
        peaks_optimal = np.zeros_like(peaks)
        for i,p in enumerate(peaks):
            peak_opt = p
            peak_opt_mass = np.sum(combined_warped[topH:botH, p - windW//2 : p + windW//2])
            for windO in range(p - deltaW, p + deltaW):
                windImg = combined_warped[topH:botH, windO - windW // 2 : windO + windW // 2]
                windMass = np.sum(windImg)
                # Penalize shift from the center a bit
                windMass = windMass - 20*abs(p - windO)
                if windMass > peak_opt_mass:
                    peak_opt_mass = windMass
                    peak_opt = windO
            peaks_optimal[i] = peak_opt


        # Mask Lines and get coordinates
        for i in range(len(peaks_optimal)):
            p = peaks_optimal[i]
            c = (255, 0, 0) if p < w/2 else (0, 255, 0)
            line_mask = np.zeros(combined_warped.shape, np.uint8)
            line_mask[topH:botH,p-windW//2:p+windW//2] = combined_warped[topH:botH,p-windW//2:p+windW//2]
            combined_warped_img[line_mask == 1] = c
            line_patch_coords = np.where(line_mask)
            if p < w/2:
                leftx = np.concatenate([leftx, line_patch_coords[1]])
                lefty = np.concatenate([lefty, line_patch_coords[0]])
            else:
                rightx = np.concatenate([rightx, line_patch_coords[1]])
                righty = np.concatenate([righty, line_patch_coords[0]])


        # Draw selected rectangles and optimal
        for i in range(len(peaks)):
            p = peaks[i]
            c = (255, 0, 0) if p < w/2 else (0, 255, 0)
            cv2.rectangle(combined_warped_img, (p-windW//2, botH), (p+windW//2, topH), c, 2)
            p = peaks_optimal[i]
            cv2.rectangle(combined_warped_img, (p-windW//2, botH), (p+windW//2, topH), c, 7)


        # ==== Prepare for the next iteration
        peaks = peaks_optimal

    # <<<<< End for loop


    # Polifit left/right lines
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)


    # Store to the latest_fit
    self.updateLatestFit(left_fit, right_fit)


    # Show: Fitted Lines
    lines_img = self.drawLines(combined_warped_img)
    compose_images(resImg, lines_img, 2, 2, 3)
    if sampled:
      save_output_img(lines_img, '02%d_fitted_lines' % self.counter)


    # Show: Lines on original undistorted image
    lines_on_road = self.drawLinesRegionOnRoad(undist)
    compose_images(resImg, lines_on_road, 2, 2, 4)
    if sampled:
      save_output_img(lines_on_road, '02%d_lines_on_road' % self.counter)


    # Add resulting calculations on the image
    self.calcCurvaturesAndCenter()
    cv2.putText(resImg, "Center Distance: %.4f m" % (self.center_distance), (int(w/2) + 20, int(h/4) + 0*40 + 45), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 2)
    cv2.putText(resImg, "Left Curvature: %.2f m" % (self.left_curverad), (int(w/2) + 20, int(h/4) + 1*40 + 45), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 2)
    cv2.putText(resImg, "Right Curvature: %.2f m" % (self.right_curverad), (int(w/2) + 20, int(h/4) + 2*40 + 45), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 255, 255], 2)


    if sampled:
        self.samples_proc.append(resImg)
        save_output_img(resImg, '02%d_result' % self.counter)


    # Increase frame counter
    self.counter += 1

    return resImg



def process_func(img):
  img_copy = np.copy(img)
  return img_copy


def main():
  parser = argparse.ArgumentParser(description="Find Lane Lines on a video")
  parser.add_argument('--video', type=str, default='project_video.mp4', help='project video')
  parser.add_argument('--output_video', default='output.mp4', type=str, help='output video')
  parser.add_argument('--mtx_dist', type=str, default='dist_pickle.p', help='saved file (for mtx and dist params)')
  parser.add_argument('--verbose', default=False, action='store_true', help='verbosity flag')
  parser.add_argument('--t_start', type=float, default=0.0, help='t_start param')
  parser.add_argument('--t_end', type=float, default=0.0, help='t_end param')

  args = parser.parse_args()

  video_file = args.video
  output_video_file = args.output_video
  mtx_dist_file = args.mtx_dist
  verbose = args.verbose
  t_start = args.t_start
  t_end = args.t_end

  print("Video file: {}".format(video_file))
  print("Output video file: {}".format(output_video_file))
  print("Mtx/Dist file: {}".format(mtx_dist_file))
  print("t_start: {}".format(t_start))
  print("t_end: {}".format(t_end))
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
  if t_end > 0.0:
    clip = clip.subclip(t_start=t_start, t_end=t_end)
  else:
    clip = clip.subclip(t_start=t_start)

  sampling = 6./(clip.duration * 25) if verbose else 0
  lineFinder = LineFinder(mtx=mtx, dist=dist, sampling=sampling)
  clip = clip.fl_image(lineFinder.process_image)
  clip.write_videofile(output_video_file, audio=False)

  # Show samples
  '''
  if verbose:
    for idx, (sample_orig, sample_proc) in enumerate(zip(lineFinder.samples_orig, lineFinder.samples_proc)):
      # print('idx=', idx)
      plt.figure(figsize=(20, 10))
      plt.subplot(1, 1, 1)
      cur_axes = plt.gca()
      cur_axes.axes.get_xaxis().set_ticks([])
      cur_axes.axes.get_yaxis().set_ticks([])
      plt.title('Processed #{}'.format(idx), fontsize=20)
      plt.imshow(sample_proc)
      plt.savefig('output_images/sample_%02d' % idx)
      # plt.show()
  '''

if __name__ == '__main__':
  main()
