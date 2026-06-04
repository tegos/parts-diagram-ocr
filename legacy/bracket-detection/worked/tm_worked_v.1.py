import cv2
import imutils as imutils
import numpy as np
from matplotlib import pyplot as plt
from functions import *
from config import *

image_path = '../res/555.png'
image_template = '../res/template.png'

img_rgb = cv2.imread(image_path)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread(image_template, 0)

img_original = img_rgb.copy()

w, h = template.shape[::-1]
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
threshold = 0.7
loc = np.where(res >= threshold)
for pt in zip(*loc[::-1]):
    bound_1 = pt
    bound_2 = (pt[0] + w, pt[1] + h)
    cv2.rectangle(img_rgb, bound_1, bound_2, (0, 0, 255), 1)
    # draw diagonal throw contour
    line_coord_x = (bound_1[0], bound_1[1] + h)
    line_coord_y = (bound_2[0], bound_1[1])
    draw_full_line(line_coord_x, line_coord_y, img_rgb)

    # detecting main digit
    # for this test direction of main digit is down 1
    direction = 1
    rect_digit_size = 40
    digit_offset = rect_digit_size * direction
    point_1_perpendicular = get_line_coord_perpendicular(line_coord_x, line_coord_y, digit_offset)
    point_2_perpendicular = get_line_coord_perpendicular(line_coord_x, line_coord_y, digit_offset, False)

    # print point_1_perpendicular
    # print point_2_perpendicular

    # cv2.rectangle(img_rgb, line_coord_x, point_2_perpendicular, (255, 0, 0), 2)

    rect = cv2.minAreaRect(
        np.array([line_coord_x,
                  line_coord_y,
                  point_1_perpendicular,
                  point_2_perpendicular
                  ], dtype=np.int32
                 )
    )

    center = rect[0]
    angle = rect[2]

    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)

    cv2.drawContours(img_rgb, [box], 0, CONTOUR_COLOR, 1)

    # get only digit contour
    mask = np.zeros_like(img_rgb)  # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, [box], 0, CONTOUR_COLOR, -1)  # Draw filled contour in mask
    out = np.zeros_like(img_rgb)  # Extract out the object and place into output image
    out[mask == 255] = img_rgb[mask == 255]

    # rotated = imutils.rotate_bound(out, -1 * angle)
    # cv2.imshow("Rotated (Correct)", rotated)

    im_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    im_at_mean = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)
    # gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edged = cv2.Canny(gray, 20, 100)
    cv2.imshow('Output im_at_mean', im_at_mean)
    out = im_at_mean

    # cv2.line(img_rgb, line_coord_x, point_1_perpendicular, (0, 205, 0), 2)
    # cv2.line(img_rgb, line_coord_y, point_2_perpendicular, (0, 205, 0), 2)
    # cv2.line(img_rgb, line_coord_y, point_2_perpendicular, (0, 205, 0), 2)
    # cv2.line(img_rgb, point_1_perpendicular, point_2_perpendicular, (0, 205, 0), 2)

    cropped_main_digit = img_original[bound_1[1]:bound_2[1], bound_1[0]:bound_2[0]]

    # cv2.imshow('cropped', cropped_main_digit)
    getDigitFromImage(out)
    # break

    # cv2.line(img_rgb, line_coord_x, line_coord_y, (0, 255, 0), 3)

cv2.namedWindow('Detected', cv2.CV_WINDOW_AUTOSIZE)
cv2.imshow('Detected', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('res.png', img_rgb)
