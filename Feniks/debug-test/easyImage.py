import cv2
import numpy as np
import threading
import time
import math


class CameraStream(threading.Thread):

    def __init__(self, resolution=(1280, 720)):
        """
        Streams camera frames for video streaming. Uses open-cv to obtain camera frames
        and threading to stream them.

            Args:
                resolution (tuple or list): resolution values in form of (X, Y, )
        """
        threading.Thread.__init__(self)
        self.camera = cv2.VideoCapture(0)
        self.camera.set(3, resolution[0])
        self.camera.set(4, resolution[1])

        self.frame = None
        self.stopped = False

        time.sleep(2)

    def run(self):
        ret, self.frame = self.camera.read()

        while ret:
            ret, new_frame = self.camera.read()
            if new_frame is not None:
                self.frame = new_frame

            if self.stopped:
                return

    def getFrame(self):
        """ Returns frame. """
        return self.frame

    def stopCamera(self):
        """ Stops camera. """
        self.stopped = True

    def closeCamera(self):
        """ Stops and closes camera. """
        self.stopCamera()
        self.camera.release()


DIM = (1280, 720)
K = np.array([[544.2342009154676, 0.0, 636.6259516335623],
              [0.0, 543.3891655875703, 339.95480782786325],
              [0.0, 0.0, 1.0]])
D = np.array([[-0.02268984933083508], [-0.09858964590105679],
              [-0.020989379824052228], [0.026037273348636328]])


"""
An easy to use image processing library that contains static methods.
Created for 2019 Teknofest competition.

    Methods:
        undistort -> numpy.ndarray
        create-mask -> numpy.ndarray
        connectedComponentAnalysis -> list
        drawRectfromStats -> None
        detectColors -> None
        circle-detection-from-stats -> None
        shade-center -> tuple
"""


def find_bottle(img, min_px_area, altitude, roll=0, pitch=0):
    thres = 50
    min_radius, max_radius = 100*altitude*0.05 + 10, 100*altitude*0.68 + 30

    mask = create_mask(img, 'g', cvt_hsv=True)
    filtered_bgr = cv2.bitwise_and(img, img, mask=mask)

    _, _, stats, _ = connectedComponentAnalysis(filtered_bgr)
    stats = stats[1:]
    idxs = np.where(stats[:, 4] >= min_px_area)
    stats = stats[idxs]

    biggest_circle = [0, 0, 0]
    for i in range(len(stats)):
        stat = stats[i]
        x_start, x_end = stat[0] - thres, stat[0] + stat[2] + thres
        y_start, y_end = stat[1] - thres, stat[1] + stat[3] + thres

        if y_start < 0:
            y_start = 0
        if x_start < 0:
            x_start = 0
        if x_end > 1280:
            x_end = 1280
        if y_end > 720:
            y_end = 720

        cv2.rectangle(img, (x_start, y_start), (x_end, y_end), (0, 0, 255), 5)

        points = np.zeros((4, 1, 2), dtype=np.float32)

        points[0, 0, 0] = x_start
        points[0, 0, 1] = y_start

        points[1, 0, 0] = x_start
        points[1, 0, 1] = y_end

        points[2, 0, 0] = x_end
        points[2, 0, 1] = y_start

        points[3, 0, 0] = x_end
        points[3, 0, 1] = y_end
        partial_img, K_n = undistortRegion(filtered_bgr, points, roll, pitch, 0)
        # cv2.imshow("partial", partial_img)

        gray_img = cv2.cvtColor(partial_img, cv2.COLOR_BGR2GRAY)
        raw_circles = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 1, 10,
                                       param1=30, param2=5, minRadius=0, maxRadius=0)

        if raw_circles is not None:
            circles = np.uint(np.around(raw_circles))
            biggest_circle = circles[0, 0]
            # bigger_circle = max(circles[0, :], key=lambda i: i[2])
            # if bigger_circle[2] > biggest_circle[2]:
            #     biggest_circle[0] = bigger_circle[0]
            #     biggest_circle[1] = bigger_circle[1]
            #     biggest_circle[2] = bigger_circle[2]

    if sum(biggest_circle) == 0:
        raise Exception("Could not detect any circle.")
    return diffVec(np.array(biggest_circle[:2]), K_n)  # np.array(biggest_circle), K_n


def rotation_matrix(roll, pitch, yaw):
    roll = roll * math.pi/180
    pitch = pitch * math.pi/180
    yaw = yaw * math.pi/180

    sina = math.sin(roll)
    cosa = math.cos(roll)
    R_roll = np.array([[1.0,        0.0,           0.0],
                       [0.0,        cosa,        -sina],
                       [0.0,        sina,         cosa]])

    sina = math.sin(pitch)
    cosa = math.cos(pitch)
    R_pitch = np.array([[cosa,          0.0,          sina],
                        [0.0,           1.0,          0.0],
                        [-sina,         0.0,          cosa]])

    sina = math.sin(yaw)
    cosa = math.cos(yaw)
    R_yaw = np.array([[cosa,         -sina,          0.0],
                      [sina,          cosa,          0.0],
                      [0.0,            0.0,          1.0]])

    return np.dot(np.dot(R_roll, R_pitch), R_yaw)


def undistortPoints(points, zoom_factor=1, roll=0, pitch=0, yaw=0):
    R_o = rotation_matrix(roll, pitch, yaw)
    R = np.linalg.inv(R_o)
    K_n = K.copy()
    K_n[0, 0] = zoom_factor * K_n[0, 0]
    K_n[1, 1] = zoom_factor * K_n[1, 1]
    P = np.dot(K_n, R[:3, :])
    points_undistort = points.copy()
    points_undistort = cv2.undistortPoints(points, K, D, points_undistort, np.eye(3), P)
    return points_undistort


def undistort(img, zoom_factor=0.6, roll=0, pitch=0, yaw=0):
    R_o = rotation_matrix(roll, pitch, yaw)
    R = np.linalg.inv(R_o)
    K_n = K.copy()
    K_n[0, 0] = zoom_factor * K_n[0, 0]
    K_n[1, 1] = zoom_factor * K_n[1, 1]
    P = np.dot(K_n, R[:3, :])
    dim1 = img.shape[:2][::-1]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), P, dim1, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def distortSearchRange(points, K_n=None, zoom_factor=0.6, roll=0, pitch=0, yaw=0):
    R_o = rotation_matrix(roll, pitch, yaw)
    R = R_o[:3, :3]
    if K_n is None:
        K_n = K.copy()
    K_n[0, 0] = zoom_factor * K_n[0, 0]
    K_n[1, 1] = zoom_factor * K_n[1, 1]
    K_new = np.dot(R, np.linalg.inv(K_n))
    points_r = points[:].copy()

    for i in range(points.shape[0]):
        vec = np.array([[points[i, 0, 0]], [points[i, 0, 1]], [1]])
        vec_r = np.dot(K_new, vec)
        points_r[i, 0, 0] = vec_r[0, 0]
        points_r[i, 0, 1] = vec_r[1, 0]
    points_distort = points.copy()
    points_distort = cv2.fisheye.distortPoints(points_r, K, D, points_distort)
    return points_distort


def undistortRegion(img, corners, roll, pitch, yaw):
    dim1 = img.shape[:2][::-1]
    corners_m = undistortPoints(corners, roll=roll, pitch=pitch, yaw=yaw)
    min_x = np.min(np.squeeze(corners_m[:, 0, 0]))
    min_y = np.min(np.squeeze(corners_m[:, 0, 1]))
    max_x = np.max(np.squeeze(corners_m[:, 0, 0]))
    max_y = np.max(np.squeeze(corners_m[:, 0, 1]))
    R_o = rotation_matrix(roll, pitch, yaw)
    R = np.linalg.inv(R_o)
    K_n = K.copy()
    K_m = K.copy()
    K_n[0, 2] -= min_x
    K_n[1, 2] -= min_y
    P = np.dot(K_n, R[:3, :])
    dim1 = (int((max_x - min_x)), int((max_y - min_y)))
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_m, D, np.eye(3), P, dim1, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
    return (undistorted_img, K_n)


def diffVec(p, K):
    vecX = np.array([[p[0]], [p[1]], [1]])
    X = np.dot(np.linalg.inv(K), vecX)
    return (X[0], X[1])


def create_mask(img, color2mask, cvt_hsv=False):
    """
    Returns an mask as numpy.ndarray to filter given color.

        Args:
            img (numpy.ndarray): source image to be masked.
            color2mask (str): 'r' for red, 'g' for green, 'b' for blue.

        Returns:
            numpy.ndarray: Mask for an image

        Raises:
            Exception: Invalid color name
    """
    if cvt_hsv:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if color2mask == 'r':  # Red
        mask1 = cv2.inRange(img, np.array([0, 100, 0]), np.array([5, 255, 255]))
        mask2 = cv2.inRange(img, np.array([160, 100, 10]), np.array([255, 255, 255]))

        mask = cv2.bitwise_or(mask1, mask2)

    elif color2mask == 'g':  # Green
        # mask = cv2.inRange(img, np.array([35, 60, 100]), np.array([80, 255, 255]))
        mask = cv2.inRange(img, np.array([40, 50, 100]), np.array([65, 100, 255]))

    elif color2mask == 'b':  # Blue
        # mask = cv2.inRange(img, np.array([80, 100, 0]), np.array([120, 255, 255]))
        mask = cv2.inRange(img, np.array([100, 150, 0]), np.array([120, 255, 180]))

    elif color2mask == 'y':  # Yellow
        mask = cv2.inRange(img, np.array([20, 70, 150]), np.array([35, 220, 255]))

    else:
        raise Exception("Invalid color name. Only r,g,b is accepted.")

    # mask = cv2.GaussianBlur(mask, (3,3), 0.1)

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    mask = cv2.medianBlur(mask, 7)
    return mask


def connectedComponentAnalysis(img):
    """
    Apply connected components analysis to given image.

        Args:
            img (numpy.ndarray): image to be applied cca.

        return:
            list: [[0]numLabels, [1]labels, [2]stats, [3]centroids]
                stats: list of 5 integers (left, top, width, height, area)
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, thres = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)

    return cv2.connectedComponentsWithStats(gray_img, connectivity=8, ltype=cv2.CV_32S)


def drawRectfromStats(image, stats, color=(0, 0, 255), thickness=3):
    """
    Draw a rectangle that surrounds connected components. Function
    uses stats of connectedComponentAnalysis.

        Args:
            image (numpy.ndarray): image that rectangle will be drawn on
            stats (list): list of 5 integers (left, top, width, height, area)
            color (tuple)=(0,0,255): color of borders as BGR (default red)
            thickness (int)=3: thickness of borders (default 3)

        return:
            None
    """
    margin = 5
    if stats[4] > 300:
        cv2.rectangle(image,
                      (stats[0]-margin, stats[1] - margin),
                      (stats[0]+stats[2]+margin, stats[1]+stats[3]+margin),
                      color,
                      thickness)


def detectColor(img, color, pixel_area):

    color_area_stats = []
    color_mask = create_mask(img, color, cvt_hsv=True)
    result = cv2.bitwise_and(img, img, mask=color_mask)
    for stats in connectedComponentAnalysis(result):
        if stats[2][4] > pixel_area:
            color_area_stats.append(stats)
    return color_area_stats


def circle_detection_from_stats(src, color, minRadius=13, maxRadius=0):

    circles = []

    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    mask = create_mask(hsv, color)
    res = cv2.bitwise_and(src, src, mask=mask)
    bgr_res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    gray_res = cv2.cvtColor(bgr_res, cv2.COLOR_BGR2GRAY)

    output_cCA = connectedComponentAnalysis(bgr_res)

    # cv2.imshow("gray_img", gray_img)
    # cv2.imshow("mask", mask)
    # cv2.imshow("res", res)
    # cv2.imshow("bgrres", bgr_res)
    # cv2.imshow("gray_res", gray_res)

    for stats in output_cCA[2][1:]:
        if stats[4] > 200:
            x_s, x_e, y_s, y_e = stats[0]-20, stats[0]+stats[2]+20, stats[1]-20, stats[1]+stats[3]+20
            try:
                cs = cv2.HoughCircles(gray_res[y_s:y_e, x_s:x_e], cv2.HOUGH_GRADIENT, 1, 50,
                                      param1=50, param2=20, minRadius=minRadius, maxRadius=maxRadius)
            except:
                continue
            if cs is not None:
                circles = np.uint16(np.around(cs))
                for circle in circles[0, :]:
                    cv2.circle(src, (circle[0]+x_s, circle[1]+y_s), circle[2], (0, 0, 255), 2)


def center_correction(src, roll_angle, pitch_angle, vision_angle=[166.9, 100]):

    pitch_angle, roll_angle = float(pitch_angle), float(roll_angle)

    dimY, dimX = src.shape[0], src.shape[1]
    nCenter_x, nCenter_y = int(dimX/2), int(dimY/2)
    shift_y, shift_x = dimX*(roll_angle/vision_angle[0]), dimY*(pitch_angle/vision_angle[1])

    return int(nCenter_x+shift_x), int(nCenter_y-shift_y)


def draw_circle(src, center, radius, color, thickness=3):
    cv2.circle(src, center, radius, color, thickness=thickness)


def bgr2hsv(bgr_img):
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
