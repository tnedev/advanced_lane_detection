import glob

from scipy import signal

import cv2
import numpy as np
from calibration import load_calib_coef, undistort_image
import matplotlib.pyplot as plt

mtx, dist = load_calib_coef()


def mask_image(img):
    """
    Mask an image with a triangle removing all data outside of the lane
    :param img:
    :return:
    """
    ysize, xsize = img.shape

    left_bottom = [0, ysize]
    right_bottom = [xsize, ysize]
    apex = [int(xsize / 2), int(ysize/1.9)]

    # This time we are defining a four sided polygon to mask
    vertices = np.array([[left_bottom, apex, apex, right_bottom]], dtype=np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)

    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def get_color_thresh(img):
    """
    Calculate a color threshold for an image
    :param img:
    :return: binary color threshold image
    """
    s_thresh = (130, 255)
    img = undistort_image(img, mtx, dist)
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hls[:, :, 2]

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    return s_binary


def get_gradient_thresh(img):
    """
    Calculate a gradient threshold of an image
    :param img: BGR image
    :return: Binary gradient threshold image
    """
    ksize = 5
    img = undistort_image(img, mtx, dist)
    #hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    #s_channel = hls[:, :, 2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(30, 150))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(30, 150))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(10, 150))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(0.5, 1.3))

    # Combines the gradient threshold images
    combined = np.zeros_like(gradx)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
    Directional threshold
    :param gray:
    :param sobel_kernel:
    :param thresh:
    :return:
    """

    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def abs_sobel_thresh(gray, orient='x', thresh=(0, 255), sobel_kernel=5):
    """
    Absolute sobel threshold
    :param gray:
    :param orient:
    :param thresh:
    :param sobel_kernel:
    :return:
    """
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Magnitude threshold
    :param gray:
    :param sobel_kernel:
    :param mag_thresh:
    :return:
    """
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


def perspective_transform(img):
    """
    Do a perspective transform over an image.
    Points are hardcoded and depend on the camera and it's positioning
    :param img:
    :return:
    """
    pts1 = np.float32([[250, 686], [1040, 680], [740, 490], [523, 492]])
    pts2 = np.float32([[295, 724], [980, 724], [988, 164], [297, 150]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_image = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

    return transformed_image


def inverse_perspective_transform(img):
    """
    Do a inverse of a perspective transform.
    :param img:
    :return:
    """
    pts2 = np.float32([[250, 686], [1040, 680], [740, 490], [523, 492]])
    pts1 = np.float32([[295, 724], [980, 724], [988, 164], [297, 150]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_image = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))

    return transformed_image


def get_thresholded_binary_image(img):
    """
    Get an image pipelined though the threshold stack
    :param img:
    :return:
    """
    color_thresh_image = get_color_thresh(img)
    grad_thresh_image = get_gradient_thresh(img)
    binary_output = np.zeros_like(grad_thresh_image)
    # Combines the gradient and color thresholds with OR
    binary_output[(grad_thresh_image == 1) | (color_thresh_image == 1)] = 1

    output_img = mask_image(binary_output)
    return output_img


def test_images():
    """
    Do processing over the test images
    :return:
    """

    test_images = glob.glob('test_images/*.jpg')

    for fname in test_images:
        print(fname)
        img = cv2.imread(fname)
        output = get_thresholded_binary_image(img)
        #cv2.imwrite(fname.replace('test_images', 'output_images/masked_images'), output)

        dst = perspective_transform(output)

        # cv2.imwrite(fname.replace('test_images', 'output_images/warped_images'), dst)
        #cv2.imshow("img", dst)

        cv2.waitKey(1000)

    cv2.destroyAllWindows()
