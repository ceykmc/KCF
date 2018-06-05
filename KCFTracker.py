# -*- coding: utf-8 -*-

import math
import numpy as np
from numpy.fft import fft2, ifft2

from utility import image_resize


# utility patch start

def compute_patch_position(object_position, padding=2.5):
    object_height = object_position[3] - object_position[1]
    object_width = object_position[2] - object_position[0]
    object_center_y = object_position[1] + object_height / 2
    object_center_x = object_position[0] + object_width / 2

    patch_height = object_height * padding
    patch_width = object_width * padding
    patch_center_y = object_center_y
    patch_center_x = object_center_x

    patch_x_1 = int(patch_center_x - patch_width / 2)
    patch_y_1 = int(patch_center_y - patch_height / 2)
    patch_x_2 = int(patch_center_x + patch_width / 2)
    patch_y_2 = int(patch_center_y + patch_height / 2)

    return [patch_x_1, patch_y_1, patch_x_2, patch_y_2]


def extract_patch_roi(image, patch_position):
    image_height, image_width = image.shape[0], image.shape[1]

    # patch 存在超出当前图像范围的情况
    top_padded = -1 * min(0, patch_position[1])
    bottom_padded = -1 * min(0, image_height - 1 - patch_position[3])
    left_padded = -1 * min(0, patch_position[0])
    right_padded = -1 * min(0, image_width - 1 - patch_position[2])

    extracted_patch_position = \
        [max(0, patch_position[0]),
         max(0, patch_position[1]),
         patch_position[2],
         patch_position[3]]

    image_padded = \
        np.pad(array=image,
               pad_width=((top_padded, bottom_padded),
                          (left_padded, right_padded),
                          (0, 0)),
               mode='edge')
    return image_padded[
           extracted_patch_position[1]:extracted_patch_position[3],
           extracted_patch_position[0]:extracted_patch_position[2], :]


def create_hanning_window(patch_height, patch_width):
    hanning_y = np.expand_dims(np.hanning(patch_height), axis=1)  # height x 1
    hanning_x = np.expand_dims(np.hanning(patch_width), axis=0)  # 1 x width
    hanning_y = hanning_y.astype(dtype=np.float32, copy=False)
    hanning_x = hanning_x.astype(dtype=np.float32, copy=False)

    hanning_widow = np.dot(hanning_y, hanning_x)
    return hanning_widow


def get_patch_feature(patch_roi):
    patch_roi = patch_roi.astype(dtype=np.float32, copy=False)
    patch_roi /= 255

    patch_height = patch_roi.shape[0]
    patch_width = patch_roi.shape[1]
    hanning_window = create_hanning_window(patch_height, patch_width)
    if patch_roi.ndim == 3:
        # hanning_window shape: m x n
        # patch_roi shape: m x n x c
        # numpy broadcasting, feature shape: m x n x c
        # patch_roi 的每一个通道和 hanning_window 依次逐元素相乘
        patch_roi = np.transpose(patch_roi, axes=(2, 0, 1))
        feature = hanning_window * patch_roi
        feature = np.transpose(feature, axes=(1, 2, 0))
    else:
        feature = hanning_window * patch_roi
    return feature


# utility patch end


def gaussian_kernel_correlation(x1, x2, sigma):
    assert len(x1.shape) >= 2 and len(x2.shape) >= 2

    c = ifft2(np.sum(
        np.conj(fft2(x1, axes=(0, 1))) * fft2(x2, axes=(0, 1)),
        axis=-1))
    c = np.fft.fftshift(c).real
    d = np.sum(x1 * x1) + np.sum(x2 * x2) - 2 * c  # d.dtype is complex
    k = np.exp(-1 * np.abs(d) / (sigma * sigma) / d.size)
    return k


def compute_alpha(x, y, sigma, regular_lambda=1e-4):
    k = gaussian_kernel_correlation(x, x, sigma)
    alpha = fft2(y) / (fft2(k) + regular_lambda)
    return alpha


def compute_response(alpha, x, z, sigma):
    k = gaussian_kernel_correlation(x, z, sigma)
    response = (ifft2(alpha * fft2(k))).real
    return response


class CKCFTracker(object):
    def __init__(self, image, object_position):
        self.object_position = None
        self.padding_ratio = 2.5
        self.patch_position = None  # object position after padding
        self.resized_patch_size = (64, 64)
        self.resized_patch_feature = None

        self.correlation_sigma = 0.2  # used in train and detect
        self.alpha = None

        self.hanning_window = self.create_hanning_window(
            self.resized_patch_size[0], self.resized_patch_size[1])
        self.training_target = self.get_training_target(
            self.resized_patch_size[0], self.resized_patch_size[1], self.padding_ratio)

        self.init(image, object_position)

    @staticmethod
    def create_hanning_window(height, width):
        hanning_y = np.expand_dims(np.hanning(height), axis=1)  # height x 1
        hanning_x = np.expand_dims(np.hanning(width), axis=0)  # 1 x width
        hanning_y = hanning_y.astype(dtype=np.float32, copy=False)
        hanning_x = hanning_x.astype(dtype=np.float32, copy=False)

        hanning_widow = np.dot(hanning_y, hanning_x)
        return hanning_widow

    @staticmethod
    def get_training_target(height, width, padding_ratio):
        mean_y, mean_x = height / 2, width / 2
        sigma = math.sqrt(height * width) / padding_ratio * 0.125
        y, x = np.ogrid[0:height, 0:width]
        return np.exp(-1 * ((y - mean_y) ** 2 + (x - mean_x) ** 2) / (2 * sigma * sigma))

    def init(self, image, object_position):
        assert image.ndim == 3

        self.object_position = object_position
        self.patch_position = compute_patch_position(object_position, self.padding_ratio)
        patch_roi = extract_patch_roi(image, self.patch_position)
        patch_roi = image_resize(patch_roi, dsize=self.resized_patch_size)
        self.resized_patch_feature = get_patch_feature(patch_roi)
        self.alpha = compute_alpha(
            self.resized_patch_feature, self.training_target, self.correlation_sigma)

    def detect(self, image):
        query_patch_roi = extract_patch_roi(image, self.patch_position)
        query_patch_roi = image_resize(query_patch_roi, dsize=self.resized_patch_size)
        query_patch_feature = get_patch_feature(query_patch_roi)
        response = compute_response(
            self.alpha, self.resized_patch_feature, query_patch_feature, self.correlation_sigma)
        max_value_location = list(np.unravel_index(np.argmax(response), response.shape))  # (y, x) format
        scale = ((self.patch_position[2] - self.patch_position[0]) / self.resized_patch_size[0],
                 (self.patch_position[3] - self.patch_position[1]) / self.resized_patch_size[1])
        offset = [(max_value_location[1] - response.shape[1] // 2) * scale[0],
                  (max_value_location[0] - response.shape[0] // 2) * scale[1]]
        x1 = min(max(0, offset[0] + self.object_position[0]), image.shape[1] - 1)
        y1 = min(max(0, offset[1] + self.object_position[1]), image.shape[0] - 1)
        x2 = min(max(0, offset[0] + self.object_position[2]), image.shape[1] - 1)
        y2 = min(max(0, offset[1] + self.object_position[3]), image.shape[0] - 1)
        detect_position = [int(e) for e in [x1, y1, x2, y2]]
        # update patch position
        self.init(image, detect_position)
        return detect_position


def main():
    pass


if __name__ == "__main__":
    main()
