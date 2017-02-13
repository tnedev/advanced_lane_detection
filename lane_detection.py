from scipy import signal

import numpy as np
import cv2
from image_processing import perspective_transform, inverse_perspective_transform, get_thresholded_binary_image
from line import Line
from moviepy.video.io.VideoFileClip import VideoFileClip


class LaneDetector:
    """
    The LaneDetector processes each video frame and outputs lane overlay over the video frames.
    It calculates the curvature of the turn and the offset of the vehicle from the center point.
    The lane detection smooths the detection over self.frames frames.
    The frame processing also validates if the detected lines are plausible.

    """
    def __init__(self):

        self.left_line = None
        self.right_line = None
        self.lane = None
        self.curvature = 0
        self.offset = 0

        # Use this many lane segments
        self.max_curvature = 7000
        self.lane_segments = 7
        self.image_offset = 100

        # Number of frames for smoothing
        self.frames = 5

        self.dists = []


    @staticmethod
    def validate_lines(left, right):
        """
        Validate if lines are parallel and withing a reasonable distance
        :param left:
        :param right:
        :return:
        """
        dist_min = 350
        dist_max = 800
        parallel_thresh = (0.0003, 0.55)
        if len(left[0]) < 3 or len(right[0]) < 3:
            return False
        else:

            new_left = Line(y=left[0], x=left[1])
            new_right = Line(y=right[0], x=right[1])

            is_parallel = new_left.is_current_fit_parallel(new_right, threshold=parallel_thresh)
            dist = new_left.get_current_fit_distance(new_right)
            is_distance_valid = dist_min < dist < dist_max
            return is_parallel & is_distance_valid

    def compare_lines(self, left_x, left_y, right_x, right_y):

        left_detected = False
        right_detected = False

        if self.validate_lines((left_x, left_y), (right_x, right_y)):
            left_detected = True
            right_detected = True
        elif self.left_line and self.right_line:
            if self.validate_lines((left_x, left_y), (self.left_line.ally, self.left_line.allx)):
                left_detected = True
            if self.validate_lines((right_x, right_y), (self.right_line.ally, self.right_line.allx)):
                right_detected = True

        return left_detected, right_detected

    def add_text_overlay(self, img):
        """
        Adds the text overlay over the frame
        :param img:
        :return:
        """
        cv2.putText(img, 'Curvature: %d m' % self.curvature, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, 'Driving %.2fm off center' % self.offset, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

    def draw_arrows(self, img):
        """
        Draws arrows pointing in the lane's direction
        :param img:
        :return:
        """

        img_height = img.shape[0]
        steps = self.lane_segments * 2
        pixels_per_step = img_height // steps

        for i in range(steps):
            start = i * pixels_per_step
            end = start + pixels_per_step

            start_point = (int(self.lane(start)), start)
            end_point = (int(self.lane(end)), end)

            if i % 2 == 0:
                img = cv2.arrowedLine(img, end_point, start_point, (255, 255, 255), 8, tipLength=0.5)

        return img

    def draw_lines(self, img, poly):
        """
        Draws the lane lines
        :param img:
        :param poly:
        :return:
        """

        img_height = img.shape[0]
        steps = self.lane_segments
        pixels_per_step = img_height // steps

        for i in range(steps):
            start = i * pixels_per_step
            end = start + pixels_per_step

            start_point = (int(poly(start)), start)
            end_point = (int(poly(end)), end)

            img = cv2.line(img, end_point, start_point, (255, 255, 255), 10)

        return img

    def draw_lane_overlay(self, img):
        """
        Draws the lane over the frame
        :param img:
        :return:
        """
        overlay = np.zeros([*img.shape])
        mask = np.zeros([img.shape[0], img.shape[1]])

        # lane area
        lane_area = self.calculate_lane_area((self.left_line, self.right_line), img.shape[0], 20)
        mask = cv2.fillPoly(mask, np.int32([lane_area]), 1)
        mask = inverse_perspective_transform(mask)

        overlay[mask == 1] = (0, 200, 0)
        selection = (overlay != 0)
        img[selection] = img[selection] * 0.3 + overlay[selection] * 0.7

        # center line
        mask[:] = 0
        mask = self.draw_arrows(mask)
        mask = inverse_perspective_transform(mask)
        img[mask == 255] = (255, 75, 2)

        # lines best
        mask[:] = 0
        mask = self.draw_lines(mask, self.left_line.best_fit_poly)
        mask = self.draw_lines(mask, self.right_line.best_fit_poly)
        mask = inverse_perspective_transform(mask)
        img[mask == 255] = (255, 200, 2)

    @staticmethod
    def get_hist_peaks(histogram, peaks, n=2, threshold=0):
        """
        Gets the peaks on the histogram
        :param n:
        :param histogram:
        :param peaks:
        :param threshold:
        :return:
        """
        if len(peaks) == 0:
            return []

        peak_list = [(peak, histogram[peak]) for peak in peaks if histogram[peak] > threshold]
        peak_list = sorted(peak_list, key=lambda x: x[1], reverse=True)

        if len(peak_list) == 0:
            return []

        x, y = zip(*peak_list)
        x = list(x)

        if len(peak_list) < n:
            return x

        return x[:n]

    def histogram_lane_detection(self, img, steps, search_window, h_window):
        all_x = []
        all_y = []
        masked_img = img[:, search_window[0]:search_window[1]]
        pixels_per_step = img.shape[0] // steps

        for i in range(steps):
            start = masked_img.shape[0] - (i * pixels_per_step)
            end = start - pixels_per_step
            histogram = np.sum(masked_img[end:start, :], axis=0)
            histogram_smooth = signal.medfilt(histogram, h_window)
            peaks = np.array(signal.find_peaks_cwt(histogram_smooth, np.arange(1, 5)))

            highest_peak = self.get_hist_peaks(histogram_smooth, peaks, n=1, threshold=5)
            if len(highest_peak) == 1:
                highest_peak = highest_peak[0]
                center = (start + end) // 2
                x, y = self.get_pixel_in_window(masked_img, highest_peak, center, pixels_per_step)

                all_x.extend(x)
                all_y.extend(y)

        all_x = np.array(all_x) + search_window[0]
        all_y = np.array(all_y)

        return all_x, all_y

    def get_windows(self, img, poly):
        """
        Get all the lane segments
        :param img:
        :param poly:
        :return:
        """

        pixels_per_step = img.shape[0] // self.lane_segments
        x_windows = []
        y_windows = []

        for i in range(self.lane_segments):
            start = img.shape[0] - (i * pixels_per_step)
            end = start - pixels_per_step

            center = (start + end) // 2
            x = poly(center)

            x, y = self.get_pixel_in_window(img, x, center, pixels_per_step)

            x_windows.extend(x)
            y_windows.extend(y)

        return x_windows, y_windows

    @staticmethod
    def get_pixel_in_window(img, x_center, y_center, size):
        """
        Get a single lane segment
        :param img:
        :param x_center:
        :param y_center:
        :param size:
        :return:
        """

        half_size = size // 2
        window = img[y_center - half_size:y_center + half_size, x_center - half_size:x_center + half_size]

        x, y = (window.T == 1).nonzero()

        x = x + x_center - half_size
        y = y + y_center - half_size

        return x, y

    def get_offset(self, frame):

        return (frame.shape[1] / 2 - self.lane(719)) * 3.7 / 700

    @staticmethod
    def calculate_lane_area(lanes, area_height, steps):

        points_left = np.zeros((steps + 1, 2))
        points_right = np.zeros((steps + 1, 2))

        for i in range(steps + 1):
            pixels_per_step = area_height // steps
            start = area_height - i * pixels_per_step

            points_left[i] = [lanes[0].best_fit_poly(start), start]
            points_right[i] = [lanes[1].best_fit_poly(start), start]

        return np.concatenate((points_left, points_right[::-1]), axis=0)

    def process_frame(self, frame):
        """
        Process a video frame
        :param frame:
        :return: The processed frame with lane and info overlay
        """
        frame_copy = np.copy(frame)

        frame = get_thresholded_binary_image(frame)
        frame = perspective_transform(frame)

        left_detected = False
        right_detected = False
        left_x = []
        left_y = []
        right_x = []
        right_y = []

        if self.left_line and self.right_line:
            left_x, left_y = self.get_windows(frame, self.left_line.best_fit_poly)
            right_x, right_y = self.get_windows(frame, self.right_line.best_fit_poly)

            left_detected, right_detected = self.compare_lines(left_x, left_y, right_x, right_y)

        if not left_detected:
            left_x, left_y = self.histogram_lane_detection(frame, self.lane_segments, (self.image_offset, frame.shape[1] // 2), h_window=7)
        if not right_detected:
            right_x, right_y = self.histogram_lane_detection(frame, self.lane_segments, (frame.shape[1] // 2, frame.shape[1] - self.image_offset), h_window=7)

        if not left_detected or not right_detected:
            left_detected, right_detected = self.compare_lines(left_x, left_y, right_x, right_y)

        # Updated left lane information.
        if left_detected:
            if self.left_line:
                self.left_line.update(y=left_x, x=left_y)
            else:
                self.left_line = Line(self.frames, left_y, left_x)

        # Updated right lane information.
        if right_detected:
            # switch x and y since lines are almost vertical
            if self.right_line:
                self.right_line.update(y=right_x, x=right_y)
            else:
                self.right_line = Line(self.frames, right_y, right_x)

        if self.left_line and self.right_line:
            self.dists.append(self.left_line.get_best_fit_distance(self.right_line))
            self.lane = (self.left_line.best_fit_poly + self.right_line.best_fit_poly) / 2
            self.curvature = Line.calc_curvature(self.lane)
            self.offset = self.get_offset(frame)

            self.draw_lane_overlay(frame_copy)
            self.add_text_overlay(frame_copy)

        cv2.imshow('lanes', frame_copy)
        cv2.waitKey(1)
        return frame_copy


video_file = "project_video.mp4"

ld = LaneDetector()

clip = VideoFileClip(video_file)
project_clip = clip.fl_image(ld.process_frame)

project_output = video_file[:-4] + '_output.mp4'
project_clip.write_videofile(project_output, audio=False)