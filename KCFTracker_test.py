# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2
from KCFTracker import CKCFTracker


def read_label_file(label_file_path, image_height, image_width):
    object_positions = list()
    with open(label_file_path, "r") as label_file:
        for line in label_file:
            c_x, c_y, w, h = [float(e) for e in line.rstrip().split(" ")[1:]]
            x1 = int((c_x - w / 2) * image_width)
            y1 = int((c_y - h / 2) * image_height)
            x2 = int((c_x + w / 2) * image_width)
            y2 = int((c_y + h / 2) * image_height)
            object_positions.append([x1, y1, x2, y2])
    return object_positions


def patch_test():
    image_path = 'test.jpg'
    label_path = 'test.txt'
    image = cv2.imread(image_path)
    object_position = read_label_file(label_path, image.shape[0], image.shape[1])[0]

    tracker = CKCFTracker(image, object_position)
    print('padding: ', tracker.padding)
    patch = tracker.patch_feature * 255
    patch = np.array(patch.astype(np.uint8))

    origin = image[object_position[1]:object_position[3],
                   object_position[0]:object_position[2], :]
    cv2.imshow('origin', origin)
    cv2.imshow('patch', patch)
    cv2.waitKey()


def permute_test():
    image_path = 'test.jpg'
    label_path = 'test.txt'
    image = cv2.imread(image_path)
    object_position = read_label_file(label_path, image.shape[0], image.shape[1])[0]
    print('src position: ', object_position)
    tracker = CKCFTracker(image, object_position)

    new_image = cv2.imread(image_path)
    height, width = new_image.shape[0], new_image.shape[1]
    # for i in range(height):
    #     new_image[i, :, :] = image[(i + 10) % height, :, :]
    for i in range(width):
        new_image[:, i, :] = image[:, (width - 15 + i) % width, :]
    dst_position = tracker.detect(new_image)
    print('dst position: ', dst_position)

    # patch position
    patch = tracker.patch_position
    cv2.rectangle(new_image, (patch[0], patch[1]), (patch[2], patch[3]), (0, 255, 0), 2)

    cv2.rectangle(image,
                  (object_position[0], object_position[1]),
                  (object_position[2], object_position[3]),
                  (0, 255, 255), 2)
    cv2.rectangle(new_image,
                  (dst_position[0], dst_position[1]),
                  (dst_position[2], dst_position[3]),
                  (0, 0, 255), 2)
    cv2.imshow('src', image)
    cv2.imshow('new', new_image)
    cv2.waitKey()


def test_folder():
    folder = r'test_data\sequence'
    image_names = [name for name in os.listdir(folder) if 'jpg' in name]
    label_names = [name.replace('jpg', 'txt') for name in image_names]

    src_image_path = os.path.join(folder, image_names[0])
    src_label_path = os.path.join(folder, label_names[0])
    src_image = cv2.imread(src_image_path)
    src_object_position = read_label_file(src_label_path, src_image.shape[0], src_image.shape[1])[0]

    tracker = CKCFTracker(src_image, src_object_position)

    for i in range(1, len(image_names)):
        image_path = os.path.join(folder, image_names[i])
        label_path = os.path.join(folder, label_names[i])
        image = cv2.imread(image_path)
        label = read_label_file(label_path, image.shape[0], image.shape[1])[0]

        detect_position = tracker.detect(image)

        cv2.rectangle(image, (label[0], label[1]), (label[2], label[3]), (0, 0, 255), 2)
        cv2.rectangle(image,
                      (detect_position[0], detect_position[1]),
                      (detect_position[2], detect_position[3]), (0, 255, 255), 2)
        cv2.imshow('detect', image)
        cv2.waitKey()


def main():
    # patch_test()
    # permute_test()
    test_folder()


if __name__ == "__main__":
    main()
