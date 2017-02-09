import numpy as np
import cv2
import glob
import pickle
import os.path

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# The board size counts just the inside edges of the chessboard, not the squares count!
board_size = (9, 6)
calib_images = glob.glob('camera_cal/*.jpg')
test_images = glob.glob('test_images/*.jpg')

def get_obj_img_points():

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in calib_images:
        img = cv2.imread(fname)
        img_org = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, board_size, corners2, ret)
            chess_board_filename = './output_images/' + fname
            print("Writing chessboard image: ", chess_board_filename)

            cv2.imwrite(chess_board_filename, img)

    return imgpoints, objpoints


def get_calib_coef():

    imgpoints, objpoints = get_obj_img_points()
    mtx = None
    dist = None

    for fname in calib_images:

        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Get the distortion matrix
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        dst = cv2.undistort(img, mtx, dist, None, mtx)

        undistort_filename = ('./output_images/' + fname)[:-4] + '_undistort.jpg'

        print("Saving undistorted chessboard image: ", undistort_filename)
        cv2.imwrite(undistort_filename, dst)

    with open('calibration.pickle', 'wb') as handle:
        pickle.dump({'mtx': mtx, 'dist': dist}, handle)

    return mtx, dist


def load_calib_coef():
    """
    Load the calibration coefficients for the camera
    From the pickle file if they are already saved or calculate them otherwise

    :return: camera matrix, distortion
    """
    if os.path.isfile('calibration.pickle'):
        with open('calibration.pickle', 'rb') as handle:
            calibration = pickle.load(handle)
            mtx = calibration.get('mtx')
            dist = calibration.get('dist')
    else:
        mtx, dist = get_calib_coef()

    return mtx, dist


def undistort_image(img, mtx, dist):
    """
    Undistort an image using this coefficients
    """
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst


def undistort_test_images():
    mtx, dist = load_calib_coef()

    for fname in test_images:
        undistorted_filename = fname.replace('test_images', 'output_images/test_images_undistorted')
        print("Saving undistorted test image: ", undistorted_filename)
        img = cv2.imread(fname)
        dst = undistort_image(img, mtx, dist)
        cv2.imwrite(undistorted_filename, dst)


# undistort_test_images()