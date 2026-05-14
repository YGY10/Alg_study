import cv2
import numpy as np


def draw_text(
    image,
    text,
    pos=(10, 30),
    scale=0.7,
    color=(0, 255, 0),
    thickness=2,
):
    cv2.putText(
        image,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return image


def make_black_image(width, height, text="NO IMAGE"):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    draw_text(image, text, pos=(20, 40), color=(0, 0, 255))
    return image


def resize_to(image, width, height):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)