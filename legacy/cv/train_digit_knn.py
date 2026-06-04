import os
import cv2
import numpy
from config import *
from functions import uniqueContour

KNN_SQUARE_SIDE_W = 20
KNN_SQUARE_SIDE_H = 20


def resize(cv_image, factor):
    new_size = tuple(map(lambda x: x * factor, cv_image.shape[::-1]))
    return cv2.resize(cv_image, new_size)


def crop(cv_image, box):
    x0, y0, x1, y1 = box
    return cv_image[y0:y1, x0:x1]


def draw_box(cv_image, box):
    x0, y0, x1, y1 = box
    cv2.rectangle(cv_image, (x0, y0), (x1, y1), (0, 0, 255), 2)


def draw_boxes_and_show(cv_image, boxes, title='N'):
    temp_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
    for box in boxes:
        croped = crop(temp_image, box)
        _, w, h = croped.shape[::-1]
        # print 'w = ', w, 'h = ', h
        draw_box(temp_image, box)
        cv2.imshow(title, temp_image)
        cv2.waitKey(0)


class BaseKnnMatcher(object):
    distance_threshold = 0

    def __init__(self, source_dir):
        self.model, self.label_map = self.get_model_and_label_map(source_dir)

    @staticmethod
    def get_model_and_label_map(source_dir):
        responses = []
        label_map = []
        samples = numpy.empty((0, KNN_SQUARE_SIDE_W * KNN_SQUARE_SIDE_H), numpy.float32)
        for label_idx, filename in enumerate(os.listdir(source_dir)):
            label = filename[:filename.index('.png')]
            label_map.append(label)
            responses.append(label_idx)

            image = cv2.imread(os.path.join(source_dir, filename), 0)

            suit_image_standard_size = cv2.resize(image, (KNN_SQUARE_SIDE_W, KNN_SQUARE_SIDE_H))

            # cv2.namedWindow('suit_image_standard_size', cv2.WINDOW_NORMAL)
            # cv2.imshow('suit_image_standard_size', suit_image_standard_size)
            # cv2.waitKey(0)

            sample = suit_image_standard_size.reshape((1, KNN_SQUARE_SIDE_W * KNN_SQUARE_SIDE_H))
            samples = numpy.append(samples, sample, 0)

        responses = numpy.array(responses, numpy.float32)
        responses = responses.reshape((responses.size, 1))
        model = cv2.KNearest()
        model.train(samples, responses)

        return model, label_map

    def predict(self, image):
        image_standard_size = cv2.resize(image, (KNN_SQUARE_SIDE_W, KNN_SQUARE_SIDE_H))

        # cv2.namedWindow('suit_image_standard_size', cv2.WINDOW_NORMAL)
        # cv2.imshow('suit_image_standard_size', image_standard_size)
        # cv2.waitKey(0)

        image_standard_size = numpy.float32(image_standard_size.reshape((1, KNN_SQUARE_SIDE_W * KNN_SQUARE_SIDE_H)))

        closest_class, results, neigh_resp, distance = self.model.find_nearest(image_standard_size, k=1)

        if distance[0][0] > self.distance_threshold:
            return None

        return self.label_map[int(closest_class)]


class DigitKnnMatcher(BaseKnnMatcher):
    distance_threshold = 10 ** 10


class MeterValueReader(object):
    def __init__(self):
        self.digit_knn_matcher = DigitKnnMatcher(source_dir='../templates')

    @classmethod
    def get_symbol_boxes(cls, cv_image):

        im_bw = cv2.GaussianBlur(cv_image, (5, 5), 0)
        cv_image = cv2.blur(cv_image, (1, 1))

        ret, thresh = cv2.threshold(cv_image.copy(), 120, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # hierarchy = hierarchy[0]

        symbol_boxes = []
        contours_new = []

        max_area = 0
        rect = []
        for ctr in contours:
            max_area = max(max_area, cv2.contourArea(ctr))

        area_ratio = 0.01

        contour_counter = 0
        for contour in contours:
            # current_hierarchy = hierarchy[contour_counter]
            # print current_hierarchy
            # print component
            inside_contour = hierarchy[0, contour_counter, 3]
            # print is_inside_contour

            x, y, width, height = cv2.boundingRect(contour)

            if cv2.contourArea(contour) > max_area * area_ratio:
                rect.append(cv2.boundingRect(cv2.approxPolyDP(contour, 1, True)))

            not_inside_contour = inside_contour < 1
            condition_for_ration_digit = (height / width) > ratio_wh

            if height > width and height > digit_height \
                    and width > min_width_digit \
                    and height < max_digit_height \
                    and not_inside_contour \
                    and condition_for_ration_digit:
                contours_new.append(contour)
                symbol_boxes.append((x, y, x + width, y + height))
            contour_counter += 1

        return symbol_boxes

    def get_value(self, meter_cv2_image):
        symbol_boxes = self.get_symbol_boxes(meter_cv2_image)
        symbol_boxes.sort()  # x is first in tuple
        symbols = []
        for box in symbol_boxes:
            symbol = self.digit_knn_matcher.predict(crop(meter_cv2_image, box))
            symbol = symbol.strip()
            symbol = symbol.strip('_')
            symbols.append(symbol)
        return symbols


if __name__ == '__main__':
    # Uncomment to generate templates from image
    # import random
    # TEMPLATE_DIR = '../templates'
    # img_bw = cv2.imread(os.path.join('../images/t.png'), 0)
    # boxes = MeterValueReader.get_symbol_boxes(img_bw)
    # for box in boxes:
    #     # You need to label templates manually after extraction
    #     cv2.imwrite(os.path.join(TEMPLATE_DIR, '%s.png' % random.randint(0, 1000)), crop(img_bw, box))

    img_bw = cv2.imread('../images/_temp/q.png', 0)
    mvr = MeterValueReader()
    print mvr.get_value(img_bw)

    boxes = mvr.get_symbol_boxes(img_bw)
    draw_boxes_and_show(img_bw, boxes)
