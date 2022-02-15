from pathlib import Path
from numpy.core.shape_base import block
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import functional as TF

from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import Image


def show_tensor_image(tensor: torch.Tensor, range_zero_one: bool = False):
    """Show a tensor of an image

    Args:
        tensor (torch.Tensor): Tensor of shape [N, 3, H, W] in range [-1, 1] or in range [0, 1]
    """
    if not range_zero_one:
        tensor = (tensor + 1) / 2
    tensor.clamp(0, 1)

    batch_size = tensor.shape[0]
    for i in range(batch_size):
        plt.title(f"Fig_{i}")
        pil_image = TF.to_pil_image(tensor[i])
        plt.imshow(pil_image)
        plt.show(block=True)


def show_editied_masked_image(
    title: str,
    source_image: Image,
    edited_image: Image,
    mask: Optional[Image] = None,
    path: Optional[Union[str, Path]] = None,
    distance: Optional[str] = None,
):
    fig_idx = 1
    rows = 1
    cols = 3 if mask is not None else 2

    fig = plt.figure(figsize=(12, 5))
    figure_title = f'Prompt: "{title}"'
    if distance is not None:
        figure_title += f" ({distance})"
    plt.title(figure_title)
    plt.axis("off")

    fig.add_subplot(rows, cols, fig_idx)
    fig_idx += 1
    _set_image_plot_name("Source Image")
    plt.imshow(source_image)

    if mask is not None:
        fig.add_subplot(rows, cols, fig_idx)
        _set_image_plot_name("Mask")
        plt.imshow(mask)
        plt.gray()
        fig_idx += 1

    fig.add_subplot(rows, cols, fig_idx)
    _set_image_plot_name("Edited Image")
    plt.imshow(edited_image)

    if path is not None:
        plt.savefig(path, bbox_inches="tight")
    else:
        plt.show(block=True)

    plt.close()


def _set_image_plot_name(name):
    plt.title(name)
    plt.xticks([])
    plt.yticks([])
