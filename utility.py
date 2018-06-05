# -*- coding: utf-8 -*-

import cv2


def image_resize(src_image, dsize=(64, 64)):
    return cv2.resize(src=src_image, dsize=dsize)


def main():
    pass


if __name__ == "__main__":
    main()
