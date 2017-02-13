import numpy as np


class Line:
    """
    Representation of a lane line

    """
    def __init__(self, frames=1, x=None, y=None):

        # Frame memory
        self.n_frames = frames
        # was the line detected in the last iteration?
        self.detected = False
        # number of pixels added per frame
        self.n_pixel_per_frame = []
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = None
        # Polynom for the current coefficients
        self.current_fit_poly = None
        # Polynom for the average coefficients over the last n iterations
        self.best_fit_poly = None
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None

        if x is not None:
            self.update(x, y)

    def update(self, x, y):

        assert len(x) == len(y), 'x and y have to be the same size'

        self.allx = x
        self.ally = y

        self.n_pixel_per_frame.append(len(self.allx))
        self.recent_xfitted.extend(self.allx)

        if len(self.n_pixel_per_frame) > self.n_frames:
            n_x_to_remove = self.n_pixel_per_frame.pop(0)
            self.recent_xfitted = self.recent_xfitted[n_x_to_remove:]

        self.bestx = np.mean(self.recent_xfitted)

        self.current_fit = np.polyfit(self.allx, self.ally, 2)

        if self.best_fit is None:
            self.best_fit = self.current_fit
        else:
            self.best_fit = (self.best_fit * (self.n_frames - 1) + self.current_fit) / self.n_frames

        self.current_fit_poly = np.poly1d(self.current_fit)
        self.best_fit_poly = np.poly1d(self.best_fit)

    def is_current_fit_parallel(self, other_line, threshold=(0, 0)):
        first_coef_dif = np.abs(self.current_fit[0] - other_line.current_fit[0])
        second_coef_dif = np.abs(self.current_fit[1] - other_line.current_fit[1])

        is_parallel = first_coef_dif < threshold[0] and second_coef_dif < threshold[1]

        return is_parallel

    def get_current_fit_distance(self, other_line):

        return np.abs(self.current_fit_poly(719) - other_line.current_fit_poly(719))

    def get_best_fit_distance(self, other_line):

        return np.abs(self.best_fit_poly(719) - other_line.best_fit_poly(719))

    @staticmethod
    def calc_curvature(fit_cr):

        ym_per_pix = 30 / 720  # meters per pixel
        xm_per_pix = 3.7 / 700  # meters per pixel

        y = np.array(np.linspace(0, 719, num=10))
        x = np.array([fit_cr(x) for x in y])
        y_eval = np.max(y)

        fit_cr = np.polyfit(y * ym_per_pix, x * xm_per_pix, 2)
        curve = ((1 + (2 * fit_cr[0] * y_eval / 2. + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])

        return curve