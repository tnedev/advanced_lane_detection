
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[image_chess]: ./camera_cal/calibration1.jpg "Chessboard Example"
[image_chess_und]: ./output_images/camera_cal/calibration1_undistort.jpg "Undistored Chessboard"
[image_chess_detect]: ./output_images/camera_cal/calibration2.jpg "Chessboard Detection"
[image_distorted]: ./test_images/test1.jpg "Distored Image"
[image_undistorted]: ./output_images/test_images_undistorted/test1.jpg "Distored Image"
[image_grad_threshold]: ./output_images/grad_thresh_images/test1.jpg "Grad Threshold"
[image_color_threshold]: ./output_images/color_thresh_images/test1.jpg "Color Threshold"
[image_threshold]: ./output_images/thresh_images/test1.jpg "Combined Threshold Image"
[image_mask]: ./output_images/mask.jpg "Mask"
[image_masked]: ./output_images/masked_images/test1.jpg "Masked Image"
[image_transform]: ./output_images/p_transform.jpg "Perspective Transform"
[image_final]: ./output_images/final_out.jpg "Final Image"

[video1]: ./project_video_output.mp4 "Video"

### README

### Camera Calibration

#### 1.Camera Calibration

First step in the process is the camera calibration. Calibration is needed because the camera lenses add distortion to the images.
Because of the distortion straight lines might end up with curvature which otherwise will add huge errors in the algorithms to follow.

The code representing the process if found under calibration.py

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Here OpenCV detects the chessboard squares:

![alt text][image_chess_detect]

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. 
Saved the camera calibration matrix and distortion coefficients into the calibration.pickle file to use when needed.


I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
### 
Original Chessboard Image

![alt text][image_chess]
### 
Undistorted Chessboard Image

![alt text][image_chess_und]

### Pipeline

#### 1. Distortion Correction
Every image or video frame is then pipelined through the distortion correction.
The effect might not seem much over those images, but could hugely affect the calculations after.
The function undistort_image in calibration.py is used.

Original Image

![alt text][image_distorted]

Distortion Corrected Image

![alt text][image_undistorted]

#### 2. Color and Gradient Threshold

Code under: image_processing.py

The goal of the second step is to provide a binary image where the lane lines are clearly visible
over wide range of road conditions. 
To achieve this we use thresholding. 

The gradient threshold uses grayscale images and detects a contrast (change) in the values.
We use a combination of sobel operations over x, y, direction and magnitude in order to include the lines information and
filter out the noise. 

The color threshold uses a pixel magnitude threshold in a color channel where lines contrast.
The color threshold is very useful on the condition where there is no good contrast between the lanes and the road.
For example yellow lines on a bright gray road will not be detected by the gradient.
By performing experiments over a few color representation I saw that the HLS color space provides a good contrast on the lines.
The information in the S channel is good for color threshold and in the combination with the gradient threshold, we will be able
to detect lines under shadows, low contrast and more.

The functions representing both techniques are found in get_color_thres and get_gradient_thresh functions.

I then combine both type of thresholds by addition.
 
The final step in this process is to apply a mask over the image in order to clear the noise.
I use a triangle mask which clears most of the information out of the lines.
The threshold pipeline is found in the get_thresholded_binary_image function.

Original Image

![alt text][image2]

Gradient Threshold Image

![alt text][image_grad_threshold]

Color Threshold Image

![alt text][image_color_threshold]

Combined Threshold Image

![alt text][image_threshold]

Image Mask

![alt text][image_mask]

Image output at the end of this step

![alt text][image_masked]

#### 3. Perspective Transform

The third step in the process is perspective transform. 
Normally, we have the camera at the windshield of the vehicle and this is our perspective.
Under this perspective our vision ann all lines in the image converge to a single distant point.
This comes from the fact that we are representing a 3D space into a 2D plane.
This also means that our lane lines do not look parallel and converge. Calculating the road curvature from those lane lines 
will be impossible. That's why we use OpenCV to transform the perspective hand have a birds eye view over the road.
By doing this our lines will be now parallel and we could easily calculate curvature. 

I have two functions perspective_transform and inverse_perspective_transform in image_processing.py to the task.

To do the perspective transform we need to select 4 point in the source and output of the transform.
This means that the input points will be warped to the output coordinates.
Because we want our lane lines to become parallel, we chose 2 points on each line of a straight road as input.
The output points we chose by roughly imaging where the points will be if the lane lines were paralel.

The inverse perspective tranform is used do draw the lane on the original frame.

Here are my points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 250, 686      | 295, 724      | 
| 1040, 680     | 980, 724      |
| 740, 490      | 988, 164      |
| 523, 492      | 297, 150      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Perspective transform output

![alt text][image_transform]

#### 4. Lane Lines Representation

In the fourth step of the process we want to take the lane lines pixel and represent as a 2nd order polynomial.

We use the Line class in lane.py to represent our left and right lines.

First part of this process is to detect the lines. 
To do this we take the birds eye view binary image and slice it into windows (get_windows method).
The histogram of each window is calculated (histogram_lane_detection, get_hist_peaks methods) and here we assume that the lane lines will form the highest peaks in the histogram.
We divide each window to left and right part and takes its histogram max as the position of the left and right lines.

After we collect the points representing the lane left and right lane lines we use them
to fit a second order polynomial. 

#### 5. Curvature and Distance Calculation

In the next step of the process we calculate the curvature of the road and the distance off center of the vehicle.

At this point we have a 2nd order polynomial representing our lines, however, those lines represent pixels and not real life
measurement. Therefore we take into account the an estimate of the road to pixel distance and 
road width in order to be able to transform the pixel space into a meter/radians space.

We calculate the curvature in the line.py calc_curvature method using the provided data. 
The offset we calculate in the get_offset method found in lane_detection.py

To calculate the offset we assume that the camera is pointing to the middle of the lane and use the pixel to 
meters conversion to calculate the offset. Negative number shows the vehicle on the left-hand side of the lane.


#### 6. Validation

We use validation and smoothing in order to get robust lines detection. 

After we detect the left and right lanes we pass the data to a validator.
The validator first checks if the lines are roughly parallel to each other. If they
are not it means that our algorithm did not detect the lines correctly and we don't use 
this data. 
The validator then checks the distance between the lines if it falls within the US standards.
If the distance is smaller or larger it again means that we did not get the correct lanes and throw away the measurement.

### Result


Here's a the final result of the process

![alt text][image_final]

### Video

#### Lines Update

With the video we do not start the lines detection process all over instead we try to fit new data
into the polynomial we already calculated. 

#### Smoothing

Because we cannot expect perfect lines detection and because there is some variations from frame to frame we use
averaging over a few frames in order to smooth the transition and make the detection robust.

#### Final Result

Final result over the provided video:

[link to my video result](./project_video_output.mp4)


### Discussion
This algorithm stack is an improvement over the previously implemented simple method for 
lane lines detection.
In here we use various threshold techniques in order to better detect the lane lines. 
Unlike in the simple method, here, we are able to detect the lane lines on low contrast environments and
under shadows. This won't be possible if we only used canny edge detection. 
Further, we manage to fit the lane lines into a 2nd order polynomial and represent curvature.
The method is made more robust through averaging over a few video frames and wrongly calculated lines are not used.This helps us
prevent glitches.

This method is camera and position dependant.
This means that every camera used has to pass through the calibration process first.
It is also dependent upon the position of the camera and would require parameters change if the position changes.

The method will fail if one of the lane lines is not there, however, this is a common thing in my country
where frequently on old roads only a center dividing line is present. Therefore, the algorithm could be improved to work
under single visible line. 

I would assume this method will also fail under bad road and weather conditions.
A wet road will add reflections which might fail the algorithm. Under heavy rain or snow we won't be able
to detect the lines. A further filtering might improve for such conditions. 

It is likely the method will fail under night conditions. The lights of the other vehicles will prevent us from detecting
the lane lines. 

In order to improve over this algorithm I would add an easier way to calibrate for camera, roads and positioning.
I would also improve the thresholds and precise data for conversion between pixel and meter space.
Experiments with different road and weather conditions are also needed. 

