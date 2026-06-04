import cv2
import imutils as imutils

from functions import *
from config import *
import glob
import time
import train_digit_knn as tdk

image_bracket_start = cv2.imread(image_bracket_start_path, 0)
image_bracket_end = cv2.imread(image_bracket_end_path, 0)

template = cv2.imread(image_template, 0)

path = '../images/*.png'
images = glob.glob(path)

# measure time executing
start = time.time()
total_images = 0

mvr = tdk.MeterValueReader()

for name in images:
    total_images += 1
    # print name
    file_name = os.path.basename(name)
    print file_name
    image_path = name
    img_rgb = cv2.imread(image_path)

    _, w_original, h_original = img_rgb.shape[::-1]

    print w_original, h_original

    if w_original > width_for_source:
        img_rgb = imutils.resize(img_rgb, width=width_for_source)

    if h_original > height_for_source:
        img_rgb = imutils.resize(img_rgb, height=height_for_source)

    # if h_original > width_for_source:
    #     img_rgb = imutils.resize(img_rgb, height=width_for_source)

    _, w_original, h_original = img_rgb.shape[::-1]
    print w_original, h_original

    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    img_original = img_rgb.copy()
    counter_all = 0
    for r_angle in range(0, 360, 30):
        print 'Angle: ', r_angle

        rotated = imutils.rotate_bound(template, r_angle)
        rotated_source = imutils.rotate_bound(img_gray, r_angle)
        rotated_source_original = imutils.rotate_bound(img_rgb, r_angle)
        rotated_source_original = rotated_source.copy()
        rotated_source_debug = rotated_source.copy()

        # rotated_source_original[np.where(rotated_source_original == [0])] = [1]

        # rotated_source_original[mask == 255] = (255, 255, 255)

        # cv2.namedWindow('rotated_source_original', cv2.WINDOW_NORMAL)
        # cv2.imshow('rotated_source_original', rotated_source_original)
        # cv2.waitKey(0)

        # cv2.imshow("Rotated", rotated)
        sp = scale_params
        for scale in np.linspace(sp[0], sp[1], sp[2])[::-1]:
            resized = imutils.resize(template, width=int(template.shape[1] * scale))
            resize_bracket_start = imutils.resize(image_bracket_start, width=int(image_bracket_start.shape[1] * scale))
            resize_bracket_end = imutils.resize(image_bracket_end, width=int(image_bracket_end.shape[1] * scale))

            w, h = resized.shape[::-1]

            res = cv2.matchTemplate(rotated_source_original, resized, cv2.TM_CCOEFF_NORMED)
            threshold = threshold_middle_bracket
            loc = np.where(res >= threshold)
            count_loc = len(zip(*loc[::-1]))
            if DEBUG:
                print 'Count:', count_loc, 'Scale:', scale
            if count_loc > 10:
                continue
            for pt in zip(*loc[::-1]):
                counter_all += 1
                bound_1 = pt
                bound_2 = (pt[0] + w, pt[1] + h)
                if DEBUG:
                    cv2.rectangle(rotated_source_debug, bound_1, bound_2, (0, 0, 255), 2)

                # bracket_line
                bracket_line_1 = bound_1
                bracket_line_2 = (bound_2[0], int(bound_2[1]) - h)

                # find start bracket
                start_bracket_point, end_bracket_point = findCoordStartEndBracket(
                    rotated_source_original, resized,
                    resize_bracket_start, resize_bracket_end,
                    bracket_line_1, bracket_line_2,
                    rotated_source_debug
                )

                # rect for main digit
                new_bound_main_digit_1 = (bound_1[0], bound_1[1] + h)
                new_bound_main_digit_2 = (bound_2[0], int(bound_2[1]) + int(2 * h))

                # cv2.rectangle(rotated_source_original, new_bound_main_digit_1, new_bound_main_digit_2, (0, 0, 255), 2)

                cropMainDigit = cropImage(rotated_source_original, new_bound_main_digit_1, new_bound_main_digit_2)

                if start_bracket_point is not None and end_bracket_point is not None:
                    bracket_with_digit = cropImage(rotated_source_original, start_bracket_point, end_bracket_point)
                else:
                    bracket_with_digit = None

                if isImage(cropMainDigit):
                    true_rotate_main_digit = imutils.rotate_bound(cropMainDigit, -r_angle)

                    # true_rotate_main_digit[np.where(true_rotate_main_digit == [0])] = [255]

                    res_file_true = '../images/res/' + file_name + '_' + str(r_angle) + '_rotate_digit_' + str(
                        counter_all) + '.png'
                    res_file = '../images/res/' + file_name + '_' + str(r_angle) + '_digit_' + str(counter_all) + '.png'
                    cv2.imwrite(res_file, cropMainDigit)
                    cv2.imwrite(res_file_true, true_rotate_main_digit)
                    full_path = os.path.dirname(__file__) + '/../images/res/' + file_name + '_' + str(
                        r_angle) + '_digit_' + str(counter_all) + '.png'
                    # print full_path

                    file_name_pad = AddBorder(res_file_true, pad=7)

                    image_pad = cv2.imread(file_name_pad, 0)

                    EdgeDetect(file_name_pad, 120, 255)

                    main_digit_arr = mvr.get_value(image_pad)
                    main_digit_text = ''.join(main_digit_arr)
                    # main_digit_text = ocr_space_file(filename=res_file_true)

                    details.append(main_digit_text)
                    boxes = mvr.get_symbol_boxes(true_rotate_main_digit)
                    tdk.draw_boxes_and_show(true_rotate_main_digit, boxes)

                    print main_digit_text

                if isImage(bracket_with_digit):
                    res_file = '../images/res/' + file_name + '_' + str(r_angle) + '_bracket_with_digit_' + str(
                        counter_all) + '.png'
                    cv2.imwrite(res_file, bracket_with_digit)

                # line
                if DEBUG:
                    bracket_points = draw_full_line(bracket_line_1, bracket_line_2, rotated_source_debug)

                res_file = '../images/res/' + file_name + '_' + str(r_angle) + '.png'
                cv2.imwrite(res_file, rotated_source_original)

                res_file_debug = '../images/res/' + file_name + '_' + str(r_angle) + '_debug.png'
                cv2.imwrite(res_file_debug, rotated_source_debug)

                # break

cv2.destroyAllWindows()

details = list(set(details))
print details

end = time.time()
elapsed = end - start
if total_images > 0:
    time_for_one_image = elapsed / total_images
    print 'Avg. time for one image: '
    secondsToStringTime(time_for_one_image)
print 'Total time: '
secondsToStringTime(elapsed)
print 'Total images: ', total_images
