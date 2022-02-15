import torch
from torch import nn
import kornia.augmentation as K


class ImageAugmentations(nn.Module):
    def __init__(self, output_size, augmentations_number, p=0.7):
        super().__init__()
        self.output_size = output_size
        self.augmentations_number = augmentations_number

        self.augmentations = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=p, padding_mode="border"),  # type: ignore
            K.RandomPerspective(0.7, p=p),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))

    def forward(self, input):
        """Extents the input batch with augmentations

        If the input is consists of images [I1, I2] the extended augmented output
        will be [I1_resized, I2_resized, I1_aug1, I2_aug1, I1_aug2, I2_aug2 ...]

        Args:
            input ([type]): input batch of shape [batch, C, H, W]

        Returns:
            updated batch: of shape [batch * augmentations_number, C, H, W]
        """
        # We want to multiply the number of images in the batch in contrast to regular augmantations
        # that do not change the number of samples in the batch)
        resized_images = self.avg_pool(input)
        resized_images = torch.tile(resized_images, dims=(self.augmentations_number, 1, 1, 1))

        batch_size = input.shape[0]
        # We want at least one non augmented image
        non_augmented_batch = resized_images[:batch_size]
        augmented_batch = self.augmentations(resized_images[batch_size:])
        updated_batch = torch.cat([non_augmented_batch, augmented_batch], dim=0)

        return updated_batch
