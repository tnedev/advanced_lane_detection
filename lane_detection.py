import glob

import cv2
import numpy as np
from calibration import load_calib_coef, undistort_image

mtx, dist = load_calib_coef()


def edit_image(img):
    pass


def get_color_thresh(img):
    s_thresh = (130, 255)
    img = undistort_image(img, mtx, dist)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    s_channel = hsv[:, :, 2]

    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    return s_binary


def get_gradient_thresh(img):

    ksize = 15
    img = undistort_image(img, mtx, dist)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
    gray = hsv[:, :, 2]
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(10, 100))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(10, 100))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(10, 100))
    dir_binary = dir_threshold(gray, sobel_kernel=15, thresh=(0.7, 1.3))
    #
    combined = np.zeros_like(gradx)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    return combined


def dir_threshold(gray, sobel_kernel=3, thresh=(0, np.pi/2)):

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


def pipeline(img):

    color_thresh_image = get_color_thresh(img)
    grad_thresh_image = get_gradient_thresh(img)
    binary_output = np.zeros_like(grad_thresh_image)
    binary_output[(grad_thresh_image == 1) | (color_thresh_image == 1)] = 255

    output_img = mask_image(binary_output)
    print(output_img.shape)
    return output_img


def mask_image(img):

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


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def test_images():
    test_images = glob.glob('test_images/*.jpg')

    for fname in test_images:
        print(fname)
        img = cv2.imread(fname)
        output = pipeline(img)
        #cv2.imwrite(fname.replace('test_images', 'output_images/masked_images'), output)
        cv2.imshow("img", output)

        cv2.waitKey(1000)
        #break

    cv2.destroyAllWindows()

test_images()
