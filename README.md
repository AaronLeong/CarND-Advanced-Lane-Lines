
# Udacity Advanced Lane Finding Project

The goal of the project is to detect and track lane lines from center camera and calculate lines curvature and vehicle position on the road.

[//]: # (Image References)

[undist_image]: ./output_images/test_undist3_comp.jpg "Undistorted"
[pipeline_undist_image]: ./output_images/pipeline/output_02719_undist.png "Undistorted pipeline image"
[pipeline_combined_image]: ./output_images/pipeline/output_02719_combined.png "Combined Thresholded image"
[pipeline_region_image]: ./output_images/pipeline/output_02719_sel_region.png "Region selection"
[pipeline_warped_image]: ./output_images/pipeline/output_02719_comb_warped.png "Combined Warped"
[pipeline_histogram_image]: ./output_images/pipeline/output_02719_histogram.png "Histogram"
[pipeline_fitedlines_image]: ./output_images/pipeline/output_02719_fitted_lines.png "Fitted Lines"
[pipeline_lines_image]: ./output_images/pipeline/output_02719_lines_on_road.png "Lines On Road"
[video1]: ./project_video.mp4 "Video"
[video_result]: https://youtu.be/DnsH4Gcqo-c "Result Video"
[video_sample]: ./output_images/result_video/output_sample.gif "Sample Video"

## Run

The following command process first 2 seconds(`--t_start 0.0 --t_end 2.0`) of the `project_video.mp4` file using camera matrix and distortion coefficients from `dist_pickle.p` that was calculated and saved previously (see below for details). Video output stored in `output.mp4` file. Verbose flag `--verbose` activates sampling of randomly selected images from the processing pipeline and store images on each step in `output_images` folder.
```
python find_lines.py --video project_video.mp4 --output_video output.mp4 --mtx_dist dist_pickle.p --t_start 0.0 --t_end 2.0 --verbose
```
You should get the result similar to this:
![Example of generated video ][video_sample]

To calculate camera matrix and distortion coefficient for your camera you can use command:
```
python camera_calibration.py --images_pattern 'camera_cal/calibration*.jpg' --save_file dist_pickle.p --nx 9 --ny 6 --verbose
```

## Detailed steps

The detailed steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


## Camera matrix and distortion coefficients

Camera calibration based on a set chess board images from [camera_cal](./camera_cal) and uses OpenCV functions `cv2.findChessboardCorners(gray, (nx,ny), None)` and `cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)`. Implementation of camera calibration in `calibration.py`.

Undistorted (corrected) images could be obtained by `cv2.undistort(image, mtx, dist, None, mtx)`. Sample result undistorted image:

![a][undist_image]

## Pipeline (single images)

### Step 1: Distortion correction
Sample of distortion correction from the pipeline:

![Undistorted pipeline sample][pipeline_undist_image]

Implementation in `process_image()` function of `LineFinder` class:
```
# Undistort
undist = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)
```

### Step 2: Use color transforms, gradients and magnitude thresholds to identify pixels of interest

For the given `project_video.mp4` problem I've found that combination of gradient in x axis together with direction and magnitude thresholds worked the best. Also I've applied a thresholding to Red and Saturation color channels and combined them all in one combined threshold image.

Below is the implementation of all thresholds and their combination:
```
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
```

Sample from pipeline:

![Combined Thresholded Image][pipeline_combined_image]


### Step 3: Perspective transform ("birds-eye view")

In order to determine curvature we need to look at the road from the "birds-eye" perspective.

Processing in `process_image()`:
```
# Transform to 'birds view'
combined_warped = self.warpTransform(combined)
```

Transform matrices are calculated only once per video because they are the same for our case:
```
def calculateTransformMatrices(self):
  if (self.M is None) or (self.Minv is None):
      self.M = cv2.getPerspectiveTransform(np.float32(self.region_vertices), np.float32(self.dest_vertices))
      self.Minv = cv2.getPerspectiveTransform(np.float32(self.dest_vertices), np.float32(self.region_vertices))

def warpTransform(self, img):
  return cv2.warpPerspective(img, self.M, (self.w, self.h), flags=cv2.INTER_LINEAR)
```

Here is the selected points for transformation which was manually selected (blue lines) for this specific video (e.g. camera setup).

![Region selected][pipeline_region_image]

Result of "birds-view" transformation:

![Combine Warped][pipeline_warped_image]

Warped original image:

![Warped Original](./output_images/pipeline/output_02719_warped_orig.png)

Selected params is the best guess that works 'almost' for all video sequence but they are not ideal because there bumps and hills on the road that change the perspective and in such situations lines deviate from the parallel.


### Step 4: Lane Pixels Identification and polynomial fit

For the first frame (or any other without prior knowledge about lines from previous frame) we use histogram of the lower part of the warped combined image to make a best guess where left and right lines start.

![Histogram Image][pipeline_histogram_image]

Then I've implemented a simple algorithm to find 2 peaks that worked better that other out of the [box solutions](https://blog.ytotech.com/2015/11/01/findpeaks-in-python/) (probably it takes more time to fine tune params of those algorithms)

```
def find_peaks(bin_img):
  histogram = np.sum(bin_img[bin_img.shape[0]/2:,:], axis=0)

  hist_ind = np.argsort(histogram)
  hist_ind = hist_ind[::-1]
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

```

Then with sliding window I've moved from bottom to the top of the picture each step adjusting window to the position with a higher density of active pixels. Prior knowledge of the line position (whether it came from histogram or from the previous window below or from the previous fitted line point) has a priority which I've coded with a slight penalty for moving window to the left or right which helped to prevent jumps in a position with slightly more active pixels and smoothed the final search result.

```
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

```

![Fitted Lines][pipeline_fitedlines_image]

Note that histogram peaks detected for each frame because it's data used for visualisation purposes. But in sliding window we are using positions of left and right lanes from previously calculated fit so we don't need the histogram data on each frame.


### Step 5: Radius of curvature and vehicle position

Given the known distances in real world like length of our selected region and lane width we can set `ym_per_pix` an `xm_per_pix` coefficients for `y` and `x` axis respectively.

Then we need to adjust our polynomial coefficients that we get for pixels given new `x` and `y` scale:
```
def scale_fit(fit, yratio, xratio):
  return np.array([fit[0]*xratio/(yratio**2), fit[1]*xratio/yratio, fit[2]*xratio])

```

Then just add the formula of curvature radius to the the problem:
```
def curve_rad(fit, y):
  return ((1 + (2*fit[0]*y + fit[1])**2)**1.5) / np.absolute(2*fit[0])
```

To calculate the position of the car from the center I assumed that a camera placed in the center of the car so we can calculate the shift as a difference from the center of the image to the center of the calculated lines multiplied by the scaled coefficient `xm_per_pix`.

The complete routine for radius of curvature calculation and shift from the center below:
```
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

  # ============ Distance from center
  center_point = (fit_getx(self.latest_right_fit, y_eval) + fit_getx(self.latest_left_fit, y_eval))/2
  self.center_distance = (self.w/2 - center_point) * xm_per_pix

```

### Step 6: Final plotting

The final visualisation of the left and right lines is made through the `cv2.fillPoly()`. Routine is below:
```
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
```

And the result that looks like:

![Lines on the Road][pipeline_lines_image]


---

## Pipeline (video)

The final video with the detected lines and all steps visualized is below:

[![Find Lanes Result](https://img.youtube.com/vi/DnsH4Gcqo-c/0.jpg)](https://www.youtube.com/watch?v=DnsH4Gcqo-c)

---

## Discussion

There a lot of things that could be done better.


### 1. Gradient thresholds and Color thresholds fail for different light/road conditions

Brief test on a challenge video showed that selected levels for thresholding are not optimal and don't give the best lines. For a general line finder I think any rigid selection of the parameter could be not optimal and the best way to adjust it's parameters automatically for any road condition. I don't know exactly how to do it, but probably some smart search or reinforcement learning algorithm with a smart policy could work here.

### 2. Rigid warp transform will fail for different camera setup

Region selection for warp transform is hardcoded and it's not scalable for different camera position//angle. Even on a road with an extensive hills (e.g. San Francisco roads) we will not have a parallel lines.

### 3. Validation of found lines (number, validness, etc)

There is no checks for situation where just one line detected or car started to change a lane and lines started moving from left to right (or right to left) and change it's location dramatically. I've just didn't test for such situation and I am pretty sure there will be problems. But for the project video `project_video.mp4` everything works smoothly.

### 4. Code structure and a better separation

This is more an engineering problem but I think there is not enough structure in my code. Some unnecessary dependency and not always clear separation between functions in a class `LineFinder` and just functions.

### 5. Next challenge: Try this code on a Raspberry Pi car

Next turn is to build a small car and test this pipeline in a [race](https://www.meetup.com/DIYRobocars/) :)
