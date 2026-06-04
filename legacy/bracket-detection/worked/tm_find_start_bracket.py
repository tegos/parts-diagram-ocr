import cv2
import imutils as imutils
from functions import *
from config import *
import glob

image_template = '../res/templ.png'
image_bracket_start_path = '../res/bracket_start.png'
image_bracket_start_path = '../res/909.png'

image_bracket_start = cv2.imread(image_bracket_start_path, 0)

template = cv2.imread(image_template, 0)

path = '../images/*.png'
images = glob.glob(path)

for name in images:
    # print name
    file_name = os.path.basename(name)
    print file_name
    image_path = name
    img_rgb = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    img_original = img_rgb.copy()
    counter_all = 0
    for r_angle in range(0, 360, 30):
        print r_angle

        rotated = imutils.rotate_bound(template, r_angle)
        rotated_source = imutils.rotate_bound(img_gray, r_angle)
        # cv2.imshow("Rotated", rotated)

        for scale in np.linspace(0.5, 1.7, 9)[::-1]:
            resized = imutils.resize(template, width=int(template.shape[1] * scale))
            resize_bracket_start = imutils.resize(image_bracket_start, width=int(image_bracket_start.shape[1] * scale))

            w, h = resized.shape[::-1]

            res = cv2.matchTemplate(rotated_source, resized, cv2.TM_CCOEFF_NORMED)
            threshold = 0.83
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                counter_all += 1
                bound_1 = pt
                bound_2 = (pt[0] + w, pt[1] + h)
                # cv2.rectangle(rotated_source, bound_1, bound_2, (0, 0, 255), 3)

                # cv2.namedWindow('Rotated', cv2.WINDOW_NORMAL)
                # cv2.imshow("Rotated", rotated_source)
                # cv2.waitKey(0)

                # draw_full_line(line_coord_x, line_coord_y, img_rgb)

                # detecting main digit
                direction = 1
                rect_digit_size = 40

                # rect for main digit
                new_bound_main_digit_1 = (bound_1[0], bound_1[1] + h)
                new_bound_main_digit_2 = (bound_2[0], int(bound_2[1]) + int(2 * h))
                cv2.rectangle(rotated_source, new_bound_main_digit_1, new_bound_main_digit_2, (0, 0, 255), 2)

                # bracket_line
                bracket_line_1 = bound_1
                bracket_line_2 = (bound_2[0], int(bound_2[1]) - h)
                bracket_points = draw_full_line(bracket_line_1, bracket_line_2, rotated_source)

                # find start bracket
                findCoordStartEndBracket(
                    rotated_source, resized, resize_bracket_start, bracket_line_1, bracket_line_2
                )
                print counter_all
                # print point_1_perpendicular
                # print point_2_perpendicular

                # cv2.rectangle(img_rgb, line_coord_x, point_2_perpendicular, (255, 0, 0), 2)
                #
                # rect = cv2.minAreaRect(
                #     np.array([line_coord_x,
                #               line_coord_y,
                #               point_1_perpendicular,
                #               point_2_perpendicular
                #               ], dtype=np.int32
                #              )
                # )
                #
                # center = rect[0]
                # angle = rect[2]

                # box = cv2.cv.BoxPoints(rect)
                # box = np.int0(box)

                # cv2.drawContours(img_rgb, [box], 0, CONTOUR_COLOR, 1)

                # get only digit contour
                mask = np.zeros_like(img_rgb)  # Create mask where white is what we want, black otherwise
                # cv2.drawContours(mask, [box], 0, CONTOUR_COLOR, -1)  # Draw filled contour in mask
                out = np.zeros_like(img_rgb)  # Extract out the object and place into output image
                out[mask == 255] = img_rgb[mask == 255]

                # rotated = imutils.rotate_bound(out, -1 * angle)
                # cv2.imshow("Rotated (Correct)", rotated)

                im_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
                gray_b = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
                # im_at_mean = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)
                # gray = cv2.GaussianBlur(gray, (3, 3), 0)
                # edged = cv2.Canny(gray, 20, 100)
                # cv2.imshow('Output im_at_mean', im_at_mean)
                # out = im_at_mean

                res_file = '../images/res/_' + str(r_angle) + file_name
                cv2.imwrite(res_file, rotated_source)

                # break

                # cv2.line(img_rgb, line_coord_x, line_coord_y, (0, 255, 0), 3)

        # cv2.namedWindow('Detected', cv2.WINDOW_NORMAL)
        # cv2.imshow('Detected', img_rgb)
        # cv2.waitKey(0)
        cv2.destroyAllWindows()
        # res_file = '../images/res/' + file_name
        # cv2.imwrite(res_file, img_rgb)
