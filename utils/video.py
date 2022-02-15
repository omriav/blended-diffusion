from typing import List
import os

from PIL.Image import Image
import cv2
import numpy as np


def save_video(images_list: List[Image], video_path: str):
    """Saves a video from a list of images

    Args:
        images_list (List[Image]): A list of PIL images.
        video_path (str): The path to save to video to.
    """
    images = [np.array(img) for img in images_list]
    height, width, _ = images[0].shape

    fps = len(images) // 20
    video = cv2.VideoWriter(video_path, 0, fps, (width, height))

    for img in images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    cv2.destroyAllWindows()
    video.release()
