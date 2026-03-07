# coding: utf-8

# Standard imports
import logging
import random
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

# External imports
import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import WeightedRandomSampler
import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split

from PIL import Image

import albumentations as A

from albumentations.pytorch import ToTensorV2

# ── Normalisation constants (computed on the plankton dataset in grayscale) ──
# Computed via scripts/normalisation/compute_norm_values.py --mode gray
# on original_dataset_optimized_512px/train
NORM_MEAN = (0.879857, 0.879857, 0.879857)
NORM_STD  = (0.190819, 0.190819, 0.190819)


def show_image(X):
    num_c = X.shape[0]
    plt.figure()
    plt.imshow(X[0] if num_c == 1 else X.permute(1, 2, 0))
    plt.show()

class AlbumentationsWrapper:
    """
    Wraps an Albumentations Compose pipeline so it can be called on a PIL Image
    (just like a torchvision transform). Converts PIL→numpy, applies augmentations,
    returns the tensor produced by ToTensorV2.
    """
    def __init__(self, album_pipeline):
        self.pipeline = album_pipeline

    def __call__(self, pil_image):
        img_np = np.array(pil_image)  # H, W, C  (RGB)
        augmented = self.pipeline(image=img_np)
        return augmented['image']     # Tensor from ToTensorV2


class AlbumentationsPreserveARWrapper:
    """
    Wraps an Albumentations pipeline with aspect-ratio-preserving preprocessing:
      1. Canonical rotation: rotate so the long side is always vertical (portrait)
      2. LongestMaxSize → scales keeping AR
      3. PadIfNeeded → pads to square with background color (white=255)
    Then applies the main augmentation pipeline.
    """
    def __init__(self, album_pipeline, target_size, pad_value=255):
        self.pipeline = album_pipeline
        self.target_size = target_size
        self.pad_value = pad_value

    def __call__(self, pil_image):
        img_np = np.array(pil_image)  # H, W, C  (RGB)
        h, w = img_np.shape[:2]

        # Step 1: Canonical rotation — long side vertical (portrait orientation)
        if w > h:
            img_np = np.rot90(img_np, k=1)  # 90° counter-clockwise

        # Step 2: Scale longest side to target_size, preserve aspect ratio
        h2, w2 = img_np.shape[:2]
        scale = self.target_size / max(h2, w2)
        new_h, new_w = int(h2 * scale), int(w2 * scale)
        import cv2
        img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

        # Step 3: Pad to square (target_size × target_size) with white background
        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        img_np = cv2.copyMakeBorder(
            img_np, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(self.pad_value, self.pad_value, self.pad_value)
        )

        # Step 4: Apply the augmentation pipeline
        augmented = self.pipeline(image=img_np)
        return augmented['image']


class PlanktonMultisourceDataset(torch.utils.data.Dataset):
    def __init__(self, sources_config, resize=224, is_validation=False):
        """
        Args:
            sources_config (dict): Dictionnary from the .yaml config file
        """
        self.sources_config = sources_config
        self.samples = []
        self.resize = resize

        ## 1. build class_to_idx (map class names to integer indices)
        # We need the original dataset to build the mapping
        if not "original_dataset" in sources_config:
            raise ValueError("sources_config must contain 'original_dataset' key")

        # Then we can check the dataset folder and build class_to_idx
        original_path = sources_config["original_dataset"]["path"]
        self.classes = self._find_classes(original_path)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        logging.info(f"Dataset initialized with {len(self.classes)} classes from the original dataset")

        ## 2. Load additional sources
        for source_name, config in sources_config.items():
            # Get the information needed in the .yaml
            root_path = config["path"]
            aug_version = config["augmentations_version"]
            weights_path = config["classes_weights_file"]

            # Get the mapping file only for non-original datasets
            if source_name != "original_dataset":
                mapping_path = config["classes_mapping_file"]
            else:
                mapping_path = None

            # Get transforms
            transforms_func = self._get_transforms(aug_version, train=(not is_validation))

            # load weights (= number of image to take in each class)
            weights = self._load_jsonc(weights_path)

            # load mapping for non-original datasets
            if mapping_path is not None:
                class_mapping = self._load_jsonc(mapping_path)
            else:
                class_mapping = {}

            count_added = 0

            for folder_name, count_limit in weights.items():
                folder_path = os.path.join(root_path, folder_name)

                if not os.path.exists(folder_path):
                    raise ValueError(f"Folder path {folder_path} does not exist")

                if source_name == "original_dataset":
                    target_class_name = folder_name
                else:
                    # Map to original class name
                    if folder_name in class_mapping:
                        target_class_name = class_mapping[folder_name]
                    else:
                        raise ValueError(f"Class {folder_name} not found in mapping for source {source_name}. Skipping.")
                        continue

                label_idx = self.class_to_idx[target_class_name]

                # List images in the folder
                images = glob.glob(os.path.join(folder_path, "*"))

                images.sort()

                images = [x for x in images if x.lower().endswith(('.png', '.jpg', '.jpeg'))]

                # Apply count limit if specified
                if count_limit != "all":
                    limit = int(count_limit)
                    random.seed(42)
                    if len(images) > limit:
                        random.shuffle(images)
                        images = images[:limit]

                # Add samples to the dataset
                for img_path in images:
                    self.samples.append((img_path, label_idx, transforms_func, source_name))
                    count_added += 1

            logging.info(f"Source [{source_name}]: {count_added} images added (Aug: {aug_version})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, transform_func, _ = self.samples[idx]
        # Conversion to RGB
        image = Image.open(path).convert("RGB") 
        
        if transform_func:
            image = transform_func(image)
        
        return image, label


    def _get_transforms(self, version_name, train=True):
        transform_list = []
        MEAN = list(NORM_MEAN)
        STD = list(NORM_STD)

        if version_name == "basic":
            # Very basic transformations
            transform_list.append(transforms.Resize((self.resize, self.resize)))

            if train:
                transform_list.extend([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(180)
                ])

            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=list(NORM_MEAN), std=list(NORM_STD)) 
            ]) 
        elif version_name == "heavy":
            # Heavy augmentations via Albumentations for better generalization
            resize = self.resize
            if train:
                album_pipeline = A.Compose([
                    A.Resize(width=resize, height=resize),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        scale=(0.85, 1.15),
                        rotate=(-180, 180),
                        translate_percent=(0, 0.08),
                        shear=(-5, 5),
                        p=0.8
                    ),
                    A.ElasticTransform(alpha=1, sigma=30, p=0.2),
                    A.CoarseDropout(
                        num_holes_range=(1, 6),
                        hole_height_range=(4, int(resize * 0.12)),
                        hole_width_range=(4, int(resize * 0.12)),
                        fill=0, p=0.5
                    ),
                    A.Downscale(scale_range=(0.75, 0.95), p=0.3),
                    A.OneOf([
                        A.MotionBlur(blur_limit=5, p=0.5),
                        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                    ], p=0.3),
                    A.OneOf([
                        A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.25), p=0.5),
                        A.MultiplicativeNoise(multiplier=(0.92, 1.08), elementwise=True, p=0.5),
                    ], p=0.4),
                    A.ImageCompression(quality_range=(65, 95), p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                    A.RandomGamma(gamma_limit=(70, 130), p=0.3),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            else:
                album_pipeline = A.Compose([
                    A.Resize(width=resize, height=resize),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            return AlbumentationsWrapper(album_pipeline)

        elif version_name == "heavy_mix":
            # Heavy augmentations optimized for use with Mixup/CutMix.
            # Slightly lighter per-image augmentations since mixing provides
            # additional regularization. Keeps RandomErasing lighter to not
            # conflict with CutMix patches.
            resize = self.resize
            if train:
                album_pipeline = A.Compose([
                    A.Resize(width=resize, height=resize),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        scale=(0.88, 1.12),
                        rotate=(-180, 180),
                        translate_percent=(0, 0.06),
                        shear=(-4, 4),
                        p=0.7
                    ),
                    A.ElasticTransform(alpha=1, sigma=25, p=0.15),
                    # Lighter CoarseDropout — CutMix already does occlusion
                    A.CoarseDropout(
                        num_holes_range=(1, 3),
                        hole_height_range=(4, int(resize * 0.08)),
                        hole_width_range=(4, int(resize * 0.08)),
                        fill=0, p=0.3
                    ),
                    A.Downscale(scale_range=(0.80, 0.95), p=0.2),
                    A.OneOf([
                        A.MotionBlur(blur_limit=5, p=0.5),
                        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                    ], p=0.25),
                    A.OneOf([
                        A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.20), p=0.5),
                        A.MultiplicativeNoise(multiplier=(0.93, 1.07), elementwise=True, p=0.5),
                    ], p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.35),
                    A.RandomGamma(gamma_limit=(75, 125), p=0.25),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            else:
                album_pipeline = A.Compose([
                    A.Resize(width=resize, height=resize),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            return AlbumentationsWrapper(album_pipeline)

        elif version_name == "dino_clean":
            # Clean geometric augmentations for foundation models (DINOv3).
            # No noise/blur/compression — the pretrained backbone is already
            # robust to those. Focus on strong spatial diversity only.
            resize = self.resize
            if train:
                album_pipeline = A.Compose([
                    A.Resize(width=resize, height=resize),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        scale=(0.85, 1.15),
                        rotate=(-180, 180),
                        translate_percent=(0, 0.08),
                        shear=(-5, 5),
                        p=0.8
                    ),
                    A.ElasticTransform(alpha=1, sigma=30, p=0.2),
                    A.CoarseDropout(
                        num_holes_range=(1, 4),
                        hole_height_range=(4, int(resize * 0.10)),
                        hole_width_range=(4, int(resize * 0.10)),
                        fill=0, p=0.4
                    ),
                    A.RandomBrightnessContrast(
                        brightness_limit=0.08, contrast_limit=0.08, p=0.3
                    ),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            else:
                album_pipeline = A.Compose([
                    A.Resize(width=resize, height=resize),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            return AlbumentationsWrapper(album_pipeline)

        elif version_name == "heavy_preserve_ar":
            # Aspect-ratio-preserving augmentations with canonical rotation.
            # The wrapper handles: landscape→portrait rotation, LongestMaxSize,
            # pad to square with white (255). The pipeline below sees a
            # (resize × resize) image already — NO A.Resize here.
            resize = self.resize
            if train:
                album_pipeline = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        scale=(0.85, 1.15),
                        rotate=(-180, 180),
                        translate_percent=(0, 0.08),
                        shear=(-5, 5),
                        p=0.8
                    ),
                    A.ElasticTransform(alpha=1, sigma=30, p=0.2),
                    A.CoarseDropout(
                        num_holes_range=(1, 6),
                        hole_height_range=(4, int(resize * 0.12)),
                        hole_width_range=(4, int(resize * 0.12)),
                        fill=255, p=0.5
                    ),
                    A.Downscale(scale_range=(0.75, 0.95), p=0.3),
                    A.OneOf([
                        A.MotionBlur(blur_limit=5, p=0.5),
                        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                    ], p=0.3),
                    A.OneOf([
                        A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.25), p=0.5),
                        A.MultiplicativeNoise(multiplier=(0.92, 1.08), elementwise=True, p=0.5),
                    ], p=0.4),
                    A.ImageCompression(quality_range=(65, 95), p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                    A.RandomGamma(gamma_limit=(70, 130), p=0.3),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            else:
                album_pipeline = A.Compose([
                    # Validation: no augmentation, just normalize
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            return AlbumentationsPreserveARWrapper(album_pipeline, target_size=resize, pad_value=255)

        elif version_name.startswith("progressive_ar_stage"):
            # ──────────────────────────────────────────────────────────
            # Progressive augmentations (AR-preserving) for fine-tuning.
            # 7 stages from very mild to extreme, all sharing the same
            # validation transform. Used by progressive_finetune().
            # ──────────────────────────────────────────────────────────
            resize = self.resize
            stage = int(version_name.split("stage")[-1])

            if not train:
                # Validation is always the same regardless of stage
                album_pipeline = A.Compose([
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
                return AlbumentationsPreserveARWrapper(album_pipeline, target_size=resize, pad_value=255)

            if stage == 0:
                # Stage 0 — WARM-UP: gentle geometry only
                # Flips + mild rotation (±45°), no noise/blur/dropout
                album_pipeline = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        scale=(0.95, 1.05),
                        rotate=(-45, 45),
                        translate_percent=(0, 0.03),
                        shear=(-2, 2),
                        p=0.6
                    ),
                    A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.2),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])

            elif stage == 1:
                # Stage 1 — LIGHT: full rotation, mild affine, light color jitter
                album_pipeline = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        scale=(0.90, 1.10),
                        rotate=(-180, 180),
                        translate_percent=(0, 0.05),
                        shear=(-3, 3),
                        p=0.7
                    ),
                    A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.3),
                    A.RandomGamma(gamma_limit=(85, 115), p=0.2),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])

            elif stage == 2:
                # Stage 2 — HARD: equivalent to heavy_preserve_ar
                album_pipeline = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        scale=(0.85, 1.15),
                        rotate=(-180, 180),
                        translate_percent=(0, 0.08),
                        shear=(-5, 5),
                        p=0.8
                    ),
                    A.ElasticTransform(alpha=1, sigma=30, p=0.2),
                    A.CoarseDropout(
                        num_holes_range=(1, 6),
                        hole_height_range=(4, int(resize * 0.12)),
                        hole_width_range=(4, int(resize * 0.12)),
                        fill=255, p=0.5
                    ),
                    A.Downscale(scale_range=(0.75, 0.95), p=0.3),
                    A.OneOf([
                        A.MotionBlur(blur_limit=5, p=0.5),
                        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                    ], p=0.3),
                    A.OneOf([
                        A.ISONoise(color_shift=(0.01, 0.03), intensity=(0.05, 0.25), p=0.5),
                        A.MultiplicativeNoise(multiplier=(0.92, 1.08), elementwise=True, p=0.5),
                    ], p=0.4),
                    A.ImageCompression(quality_range=(65, 95), p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                    A.RandomGamma(gamma_limit=(70, 130), p=0.3),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])

            elif stage == 3:
                # Stage 3 — MEDIUM-HARD: heavy_preserve_ar + first spatial distortions
                # Introduces GridDistortion, Perspective, slightly wider affine,
                # and first touch of GaussNoise. Bridge between baseline and hard.
                album_pipeline = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        scale=(0.82, 1.18),
                        rotate=(-180, 180),
                        translate_percent=(0, 0.09),
                        shear=(-6, 6),
                        p=0.82
                    ),
                    A.ElasticTransform(alpha=1.2, sigma=32, p=0.25),
                    A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.2),
                    A.Perspective(scale=(0.03, 0.07), p=0.2),
                    A.CoarseDropout(
                        num_holes_range=(1, 7),
                        hole_height_range=(4, int(resize * 0.14)),
                        hole_width_range=(4, int(resize * 0.14)),
                        fill=255, p=0.55
                    ),
                    A.Downscale(scale_range=(0.70, 0.92), p=0.35),
                    A.OneOf([
                        A.MotionBlur(blur_limit=7, p=0.5),
                        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    ], p=0.35),
                    A.OneOf([
                        A.ISONoise(color_shift=(0.01, 0.04), intensity=(0.08, 0.30), p=0.4),
                        A.MultiplicativeNoise(multiplier=(0.90, 1.10), elementwise=True, p=0.3),
                        A.GaussNoise(std_range=(0.02, 0.07), p=0.3),
                    ], p=0.45),
                    A.ImageCompression(quality_range=(55, 92), p=0.35),
                    A.RandomBrightnessContrast(brightness_limit=0.22, contrast_limit=0.25, p=0.45),
                    A.RandomGamma(gamma_limit=(65, 140), p=0.35),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])

            elif stage == 4:
                # Stage 4 — HARD: adds OpticalDistortion, Morphological, Posterize
                # All distortion types present, moderate erasure, broader intensity
                album_pipeline = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        scale=(0.78, 1.22),
                        rotate=(-180, 180),
                        translate_percent=(0, 0.10),
                        shear=(-8, 8),
                        p=0.85
                    ),
                    A.OneOf([
                        A.ElasticTransform(alpha=1.5, sigma=35, p=0.4),
                        A.GridDistortion(num_steps=6, distort_limit=0.3, p=0.3),
                        A.OpticalDistortion(distort_limit=0.2, p=0.3),
                    ], p=0.35),
                    A.Perspective(scale=(0.04, 0.09), p=0.25),
                    A.CoarseDropout(
                        num_holes_range=(2, 8),
                        hole_height_range=(5, int(resize * 0.15)),
                        hole_width_range=(5, int(resize * 0.15)),
                        fill=255, p=0.6
                    ),
                    A.Downscale(scale_range=(0.60, 0.88), p=0.4),
                    A.OneOf([
                        A.MotionBlur(blur_limit=7, p=0.4),
                        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                        A.MedianBlur(blur_limit=5, p=0.3),
                    ], p=0.4),
                    A.OneOf([
                        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.08, 0.32), p=0.4),
                        A.MultiplicativeNoise(multiplier=(0.88, 1.12), elementwise=True, p=0.3),
                        A.GaussNoise(std_range=(0.02, 0.09), p=0.3),
                    ], p=0.5),
                    A.ImageCompression(quality_range=(45, 88), p=0.4),
                    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.30, p=0.5),
                    A.RandomGamma(gamma_limit=(60, 145), p=0.38),
                    A.OneOf([
                        A.Posterize(num_bits=(5, 7), p=0.5),
                        A.Sharpen(alpha=(0.15, 0.4), lightness=(0.6, 1.0), p=0.5),
                    ], p=0.2),
                    A.Morphological(scale=(2, 3), operation="erosion", p=0.1),
                    A.Morphological(scale=(2, 3), operation="dilation", p=0.1),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])

            elif stage == 5:
                # Stage 5 — VERY HARD: well beyond heavy_preserve_ar
                # Strong spatial distortions + aggressive noise/blur + large erasure
                album_pipeline = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        scale=(0.70, 1.30),
                        rotate=(-180, 180),
                        translate_percent=(0, 0.12),
                        shear=(-10, 10),
                        p=0.9
                    ),
                    # Heavy spatial distortions
                    A.OneOf([
                        A.ElasticTransform(alpha=2, sigma=40, p=0.5),
                        A.GridDistortion(num_steps=7, distort_limit=0.4, p=0.5),
                    ], p=0.45),
                    A.OpticalDistortion(distort_limit=0.3, p=0.3),
                    A.Perspective(scale=(0.05, 0.12), p=0.3),
                    # Aggressive erasure — up to ~20% of the image
                    A.CoarseDropout(
                        num_holes_range=(3, 10),
                        hole_height_range=(6, int(resize * 0.18)),
                        hole_width_range=(6, int(resize * 0.18)),
                        fill=255, p=0.65
                    ),
                    # Heavy degradation
                    A.Downscale(scale_range=(0.50, 0.85), p=0.45),
                    A.OneOf([
                        A.MotionBlur(blur_limit=9, p=0.4),
                        A.GaussianBlur(blur_limit=(3, 9), p=0.3),
                        A.MedianBlur(blur_limit=7, p=0.3),
                    ], p=0.5),
                    # Strong noise
                    A.OneOf([
                        A.ISONoise(color_shift=(0.02, 0.06), intensity=(0.10, 0.40), p=0.4),
                        A.MultiplicativeNoise(multiplier=(0.85, 1.15), elementwise=True, p=0.3),
                        A.GaussNoise(std_range=(0.03, 0.12), p=0.3),
                    ], p=0.6),
                    A.ImageCompression(quality_range=(35, 85), p=0.45),
                    # Intensity transforms
                    A.RandomBrightnessContrast(brightness_limit=0.30, contrast_limit=0.35, p=0.55),
                    A.RandomGamma(gamma_limit=(55, 150), p=0.4),
                    A.OneOf([
                        A.Posterize(num_bits=(4, 6), p=0.5),
                        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
                    ], p=0.3),
                    # Morphological transforms (erosion/dilation)
                    A.Morphological(scale=(2, 4), operation="erosion", p=0.15),
                    A.Morphological(scale=(2, 4), operation="dilation", p=0.15),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])

            elif stage == 6:
                # Stage 6 — EXTREME / BRUTAL
                # Pushes augmentations to the limit. The model must learn to
                # classify from heavily degraded, distorted, partially occluded
                # images. Forces robustness to worst-case inputs.
                album_pipeline = A.Compose([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        scale=(0.60, 1.40),
                        rotate=(-180, 180),
                        translate_percent=(0, 0.15),
                        shear=(-15, 15),
                        p=0.95
                    ),
                    # Extreme spatial distortions — always at least one
                    A.OneOf([
                        A.ElasticTransform(alpha=3, sigma=50, p=0.4),
                        A.GridDistortion(num_steps=8, distort_limit=0.5, p=0.3),
                        A.OpticalDistortion(distort_limit=0.5, p=0.3),
                    ], p=0.6),
                    A.Perspective(scale=(0.06, 0.15), p=0.4),
                    # Massive erasure — up to ~30% of the image
                    A.CoarseDropout(
                        num_holes_range=(4, 15),
                        hole_height_range=(8, int(resize * 0.22)),
                        hole_width_range=(8, int(resize * 0.22)),
                        fill=255, p=0.75
                    ),
                    # Additional rectangular cutout (simulates large occlusion)
                    A.RandomSizedCrop(
                        min_max_height=(int(resize * 0.5), int(resize * 0.9)),
                        size=(resize, resize),
                        p=0.25
                    ),
                    # Extreme degradation
                    A.Downscale(scale_range=(0.35, 0.75), p=0.55),
                    A.OneOf([
                        A.MotionBlur(blur_limit=13, p=0.35),
                        A.GaussianBlur(blur_limit=(5, 13), p=0.30),
                        A.MedianBlur(blur_limit=9, p=0.20),
                        A.Defocus(radius=(3, 8), alias_blur=(0.1, 0.5), p=0.15),
                    ], p=0.6),
                    # Extreme noise
                    A.OneOf([
                        A.ISONoise(color_shift=(0.03, 0.08), intensity=(0.15, 0.50), p=0.35),
                        A.MultiplicativeNoise(multiplier=(0.80, 1.20), elementwise=True, p=0.30),
                        A.GaussNoise(std_range=(0.05, 0.18), p=0.35),
                    ], p=0.7),
                    A.ImageCompression(quality_range=(20, 75), p=0.55),
                    # Extreme intensity transforms
                    A.RandomBrightnessContrast(brightness_limit=0.40, contrast_limit=0.45, p=0.65),
                    A.RandomGamma(gamma_limit=(40, 180), p=0.5),
                    A.OneOf([
                        A.Posterize(num_bits=(3, 5), p=0.35),
                        A.Solarize(threshold_range=(0.5, 0.9), p=0.30),
                        A.Sharpen(alpha=(0.3, 0.7), lightness=(0.3, 1.2), p=0.35),
                    ], p=0.4),
                    # Morphological transforms (erosion/dilation) — stronger
                    A.OneOf([
                        A.Morphological(scale=(3, 6), operation="erosion", p=0.5),
                        A.Morphological(scale=(3, 6), operation="dilation", p=0.5),
                    ], p=0.3),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            else:
                raise ValueError(f"Unknown progressive stage: {stage}. Valid: 0-6.")

            return AlbumentationsPreserveARWrapper(album_pipeline, target_size=resize, pad_value=255)

        elif version_name == "ultra_heavy":
            # ULTRA HEAVY: combines all distortion families at maximum intensity.
            # Morphological + GridDistortion + OpticalDistortion + heavy noise/blur
            # + CLAHE + Solarize + Posterize + massive CoarseDropout
            resize = self.resize
            if train:
                album_pipeline = A.Compose([
                    A.Resize(width=resize, height=resize),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        scale=(0.65, 1.35),
                        rotate=(-180, 180),
                        translate_percent=(0, 0.12),
                        shear=(-12, 12),
                        p=0.9
                    ),
                    # All spatial distortions stacked
                    A.OneOf([
                        A.ElasticTransform(alpha=2.5, sigma=45, p=0.4),
                        A.GridDistortion(num_steps=7, distort_limit=0.4, p=0.3),
                        A.OpticalDistortion(distort_limit=0.4, p=0.3),
                    ], p=0.5),
                    A.Perspective(scale=(0.05, 0.12), p=0.3),
                    # Heavy erasure
                    A.CoarseDropout(
                        num_holes_range=(3, 12),
                        hole_height_range=(6, int(resize * 0.20)),
                        hole_width_range=(6, int(resize * 0.20)),
                        fill=0, p=0.7
                    ),
                    # Extreme degradation
                    A.Downscale(scale_range=(0.40, 0.80), p=0.5),
                    A.OneOf([
                        A.MotionBlur(blur_limit=11, p=0.35),
                        A.GaussianBlur(blur_limit=(3, 11), p=0.30),
                        A.MedianBlur(blur_limit=7, p=0.20),
                        A.Defocus(radius=(3, 7), alias_blur=(0.1, 0.5), p=0.15),
                    ], p=0.55),
                    # Heavy noise
                    A.OneOf([
                        A.ISONoise(color_shift=(0.02, 0.07), intensity=(0.12, 0.45), p=0.4),
                        A.MultiplicativeNoise(multiplier=(0.82, 1.18), elementwise=True, p=0.3),
                        A.GaussNoise(std_range=(0.04, 0.15), p=0.3),
                    ], p=0.65),
                    A.ImageCompression(quality_range=(25, 80), p=0.5),
                    # Heavy photometric
                    A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.40, p=0.6),
                    A.RandomGamma(gamma_limit=(45, 170), p=0.45),
                    A.CLAHE(clip_limit=(2, 6), tile_grid_size=(4, 4), p=0.3),
                    A.OneOf([
                        A.Posterize(num_bits=(3, 6), p=0.4),
                        A.Solarize(threshold_range=(0.5, 0.9), p=0.3),
                        A.Sharpen(alpha=(0.25, 0.6), lightness=(0.4, 1.1), p=0.3),
                    ], p=0.35),
                    # Morphological
                    A.OneOf([
                        A.Morphological(scale=(2, 5), operation="erosion", p=0.5),
                        A.Morphological(scale=(2, 5), operation="dilation", p=0.5),
                    ], p=0.25),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            else:
                album_pipeline = A.Compose([
                    A.Resize(width=resize, height=resize),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            return AlbumentationsWrapper(album_pipeline)

        elif version_name == "heavy_cutout":
            # Heavy CUTOUT focus: massive occlusion + GridDropout + RandomSizedCrop
            # Forces model to classify from partial information
            resize = self.resize
            if train:
                album_pipeline = A.Compose([
                    A.Resize(width=resize, height=resize),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        scale=(0.85, 1.15),
                        rotate=(-180, 180),
                        translate_percent=(0, 0.08),
                        shear=(-5, 5),
                        p=0.8
                    ),
                    # Multiple erasure strategies stacked
                    A.CoarseDropout(
                        num_holes_range=(4, 15),
                        hole_height_range=(8, int(resize * 0.22)),
                        hole_width_range=(8, int(resize * 0.22)),
                        fill=0, p=0.8
                    ),
                    A.GridDropout(ratio=0.3, random_offset=True, holes_number_xy=(4, 4), p=0.4),
                    A.RandomSizedCrop(
                        min_max_height=(int(resize * 0.5), int(resize * 0.85)),
                        size=(resize, resize),
                        p=0.3
                    ),
                    # Light noise/blur so the focus stays on occlusion
                    A.OneOf([
                        A.GaussianBlur(blur_limit=(3, 5), p=0.5),
                        A.MotionBlur(blur_limit=5, p=0.5),
                    ], p=0.25),
                    A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.35),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            else:
                album_pipeline = A.Compose([
                    A.Resize(width=resize, height=resize),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            return AlbumentationsWrapper(album_pipeline)

        elif version_name == "heavy_color":
            # Heavy PHOTOMETRIC focus: CLAHE + Equalize + Solarize + extreme
            # brightness/contrast/gamma. Forces invariance to lighting conditions.
            resize = self.resize
            if train:
                album_pipeline = A.Compose([
                    A.Resize(width=resize, height=resize),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        scale=(0.85, 1.15),
                        rotate=(-180, 180),
                        translate_percent=(0, 0.08),
                        shear=(-5, 5),
                        p=0.8
                    ),
                    # Heavy photometric stack
                    A.CLAHE(clip_limit=(2, 8), tile_grid_size=(4, 4), p=0.5),
                    A.Equalize(p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.40, contrast_limit=0.45, p=0.7),
                    A.RandomGamma(gamma_limit=(40, 180), p=0.5),
                    A.OneOf([
                        A.Solarize(threshold_range=(0.4, 0.9), p=0.4),
                        A.Posterize(num_bits=(3, 6), p=0.3),
                        A.Sharpen(alpha=(0.3, 0.7), lightness=(0.3, 1.2), p=0.3),
                    ], p=0.5),
                    A.OneOf([
                        A.ISONoise(color_shift=(0.02, 0.06), intensity=(0.10, 0.40), p=0.4),
                        A.MultiplicativeNoise(multiplier=(0.85, 1.15), elementwise=True, p=0.3),
                        A.GaussNoise(std_range=(0.03, 0.12), p=0.3),
                    ], p=0.5),
                    A.ImageCompression(quality_range=(30, 85), p=0.4),
                    # Light erasure
                    A.CoarseDropout(
                        num_holes_range=(1, 4),
                        hole_height_range=(4, int(resize * 0.10)),
                        hole_width_range=(4, int(resize * 0.10)),
                        fill=0, p=0.4
                    ),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            else:
                album_pipeline = A.Compose([
                    A.Resize(width=resize, height=resize),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            return AlbumentationsWrapper(album_pipeline)

        elif version_name == "heavy_spatial":
            # Heavy SPATIAL DISTORTION focus: ElasticTransform, GridDistortion,
            # OpticalDistortion, Perspective, PiecewiseAffine-like. Forces
            # invariance to morphological deformations.
            resize = self.resize
            if train:
                album_pipeline = A.Compose([
                    A.Resize(width=resize, height=resize),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Affine(
                        scale=(0.70, 1.30),
                        rotate=(-180, 180),
                        translate_percent=(0, 0.12),
                        shear=(-12, 12),
                        p=0.9
                    ),
                    # All spatial distortions at high intensity
                    A.ElasticTransform(alpha=2.5, sigma=40, p=0.4),
                    A.GridDistortion(num_steps=7, distort_limit=0.45, p=0.4),
                    A.OpticalDistortion(distort_limit=0.4, p=0.3),
                    A.Perspective(scale=(0.06, 0.14), p=0.35),
                    # Moderate erasure
                    A.CoarseDropout(
                        num_holes_range=(1, 6),
                        hole_height_range=(4, int(resize * 0.12)),
                        hole_width_range=(4, int(resize * 0.12)),
                        fill=0, p=0.5
                    ),
                    # Light noise/degradation only
                    A.Downscale(scale_range=(0.65, 0.90), p=0.3),
                    A.OneOf([
                        A.MotionBlur(blur_limit=7, p=0.5),
                        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
                    ], p=0.3),
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            else:
                album_pipeline = A.Compose([
                    A.Resize(width=resize, height=resize),
                    A.ToGray(p=1.0),
                    A.Normalize(mean=NORM_MEAN, std=NORM_STD),
                    ToTensorV2()
                ])
            return AlbumentationsWrapper(album_pipeline)

        elif version_name == "pelgas_basic":
            transform_list.append(transforms.Resize((self.resize, self.resize)))

            if train:
                transform_list.extend([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomVerticalFlip(p=0.5),
                    transforms.RandomRotation(180),
                ])

                transform_list.extend([
                    transforms.RandomAffine(
                        degrees=0, 
                        translate=(0.1, 0.1), 
                        scale=(0.8, 1.2), 
                        shear=10,
                        fill=0.78
                    ),
                    
                    transforms.RandomApply([
                        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))
                    ], p=0.3),
                    
                    transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
                    
                    transforms.ColorJitter(
                        brightness=0.3, 
                        contrast=0.3, 
                        saturation=0.2, 
                        hue=0.05
                    ),
                ])

            # 4. Conversion Tensor
            transform_list.append(transforms.ToTensor())
            # 6. Normalisation finale
            transform_list.append(transforms.Normalize(mean=list(NORM_MEAN), std=list(NORM_STD)))
            
        return transforms.Compose(transform_list)
    
    def _find_classes(self, dir_path):
        """Scan original folder to find class names."""
        classes = [d.name for d in os.scandir(dir_path) if d.is_dir()]
        classes.sort()
        return classes
    
    def _load_jsonc(self, filepath):
        """
        Source @gemini
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            clean_lines = [line.split('//')[0] for line in lines]
            clean_content = '\n'.join(clean_lines)
            return json.loads(clean_content)
        except Exception as e:
            logging.error(f"Error while reading JSONC for {filepath} : {e}")
            return {}

import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset

class PlanktonMultiChannelDataset(Dataset):
    def __init__(self, root_original, root_clahe, root_masks, transform=None, is_test=False, mask_ext='.png'):
        """
        Args:
            root_original (str): Chemin vers le dossier des images originales (RGB/Gris)
            root_clahe (str): Chemin vers le dossier des images CLAHE
            root_masks (str): Chemin vers le dossier des masques binaires
            transform (callable, optional): Transformation Torchvision à appliquer.
            is_test (bool): Si True, le dataset ne s'attend pas à des sous-dossiers de classes.
            mask_ext (str): Extension des fichiers masques (ex: '.png').
        """
        self.root_original = root_original
        self.root_clahe = root_clahe
        self.root_masks = root_masks
        self.transform = transform
        self.is_test = is_test
        self.mask_ext = mask_ext

        self.image_paths = []
        self.classes = []
        self.class_to_idx = {}

        # 1. Détection des images et des classes
        if not self.is_test:
            # Mode Entraînement : Structure dossier/classe/image.jpg
            self.classes = sorted([d for d in os.listdir(root_original) if os.path.isdir(os.path.join(root_original, d))])
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            
            for cls_name in self.classes:
                cls_folder = os.path.join(root_original, cls_name)
                # On prend toutes les extensions courantes
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']:
                    self.image_paths.extend(glob.glob(os.path.join(cls_folder, ext)))
        else:
            # Mode Test : Structure dossier/image.jpg (à plat)
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif']:
                self.image_paths.extend(glob.glob(os.path.join(root_original, ext)))
        
        self.image_paths = sorted(self.image_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Récupération du chemin de l'image originale
        original_path = self.image_paths[idx]
        
        # 2. Construction des chemins correspondants (CLAHE et Masque)
        # On utilise os.path.relpath pour garder la structure (ex: 'classe_A/img1.jpg')
        rel_path = os.path.relpath(original_path, self.root_original)
        
        clahe_path = os.path.join(self.root_clahe, rel_path)
        
        # Pour le masque, on doit gérer le changement d'extension (souvent .png pour les masques)
        path_no_ext = os.path.splitext(rel_path)[0]
        mask_path = os.path.join(self.root_masks, path_no_ext + self.mask_ext)

        # 3. Chargement des images avec PIL
        # .convert('L') force le chargement en niveau de gris (1 canal)
        try:
            img_orig = Image.open(original_path).convert('L')
            img_clahe = Image.open(clahe_path).convert('L')
            img_mask = Image.open(mask_path).convert('L')
        except FileNotFoundError as e:
            # Utile pour le debug si les structures de dossiers ne correspondent pas parfaitement
            raise FileNotFoundError(f"Fichier manquant pour l'index {idx}. Erreur: {e}")

        # 4. Fusion des canaux (Channel Stacking)
        # R = Original, G = CLAHE, B = Masque
        # Cela crée une image PIL mode 'RGB' standard
        combined_image = Image.merge('RGB', (img_orig, img_clahe, img_mask))

        # 5. Application des transformations Torchvision
        if self.transform:
            combined_image = self.transform(combined_image)

        # 6. Retour des données
        if self.is_test:
            # En test, on retourne l'image et son nom de fichier (pour faire le fichier de soumission)
            filename = os.path.basename(original_path)
            return combined_image, filename
        else:
            # En train/valid, on retourne l'image et son label (int)
            parent_dir = os.path.dirname(original_path)
            class_name = os.path.basename(parent_dir)
            label = self.class_to_idx[class_name]
            return combined_image, label



def identity_fn(x):
    """Fonction qui ne fait rien, remplace lambda x: x pour que Windows soit content."""
    return x


def get_train_transforms(data_config, aug_type="al", ref_images=None):
    resize = data_config["resize"]

    if aug_type == "al":

        return A.Compose([
            # --- 1. GÉOMÉTRIE ---
            A.Resize(width=resize, height=resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_CONSTANT), 
            
            # Déformation élastique légère (simule le mouvement de l'eau/corps mous)
            A.ElasticTransform(alpha=1, sigma=30, p=0.3), 

            # --- 2. ARTEFACTS (NOUVEAU) ---
            # Simule des poussières / saletés sur l'objectif ou dans l'eau
            # On crée entre 1 et 15 petits points noirs (ou gris foncé)
            A.CoarseDropout(
                num_holes_range=(2, 8),
                hole_height_range=(2, 8),
                hole_width_range=(2, 8),
                fill=0, p=0.6
            ),

            # --- 3. DÉGRADATION & BRUIT ---
            
            # Réduction de résolution (simule une optique bas de gamme)
            A.Downscale(scale_range=(0.8, 0.95), p=0.4),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.5),   
                A.GaussianBlur(blur_limit=(3, 3), p=0.5), 
            ], p=0.3),
            
            # Bruit de capteur (Remplacement de GaussNoise pour éviter l'erreur)
            A.OneOf([
                # Simulation de bruit ISO (grain photo) - très réaliste
                A.ISONoise(color_shift=(0.01, 0.02), intensity=(0.05, 0.3), p=0.5),
                
                # Bruit multiplicatif (variation de sensibilité locale)
                A.MultiplicativeNoise(multiplier=(0.95, 1.05), elementwise=True, p=0.5),
            ], p=0.5),

            # Compression JPEG (Artefacts de blocs carrés)
            A.ImageCompression(quality_range=(70, 95), p=0.4),

            # --- 4. LUMIÈRE ---
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            
            # --- 5. FINITION : FORCER LE GRIS ---
            # Convertit tout (y compris le bruit coloré ISO) en gris pur
            A.ToGray(p=1.0),
        
            A.Normalize(mean=NORM_MEAN, std=NORM_STD), 
            ToTensorV2() 
        ])

    elif aug_type == "basic":
        return A.Compose([
            A.Resize(width=resize, height=resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_CONSTANT), 
            A.ToGray(p=1.0),
        
            A.Normalize(mean=NORM_MEAN, std=NORM_STD), 
            ToTensorV2()
        ])
    
    elif aug_type == "tl_basic":
        return A.Compose([
            A.Resize(width=resize, height=resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_CONSTANT), 

            A.ToGray(p=1.0),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD), 
            ToTensorV2()
        ])

    elif aug_type == "tl_occlusion":
        return A.Compose([
            A.Resize(width=resize, height=resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, p=0.7, border_mode=cv2.BORDER_CONSTANT), 

            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(int(resize * 0.1), int(resize * 0.3)),
                hole_width_range=(int(resize * 0.1), int(resize * 0.3)),
                p=0.3
            ),

            A.ToGray(p=1.0),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD), 
            ToTensorV2()
        ])
    
    elif aug_type == "tl_occlusion_affine":
        return A.Compose([
            A.Resize(width=resize, height=resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            A.Affine(
                scale=(0.90, 1.10),      # Zoom in/out by 80-120%
                rotate=(-180, 180),      # Rotate by -15 to +15 degrees
                translate_percent=(0, 0.05), # translate by 0-10%
                shear=(-2, 2),          # shear by -10 to +10 degrees
                p=0.8
            ),

            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(int(resize * 0.1), int(resize * 0.3)),
                hole_width_range=(int(resize * 0.1), int(resize * 0.3)),
                p=0.3
            ),

            A.ToGray(p=1.0),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD), 
            ToTensorV2()
        ])
    
    elif aug_type == "tl_occlusion_affine_domain":
        return A.Compose([
            A.Resize(width=resize, height=resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            A.Affine(
                scale=(0.90, 1.10),      # Zoom in/out by 80-120%
                rotate=(-180, 180),      # Rotate by -15 to +15 degrees
                translate_percent=(0, 0.05), # translate by 0-10%
                shear=(-2, 2),          # shear by -10 to +10 degrees
                p=0.8
            ),

            A.GridDistortion(num_steps=5,
                             distort_limit=[-0.30, 0.30]),
            A.RandomGamma(gamma_limit=[40, 150], p=0.5),
            A.ISONoise(color_shift=[0.02, 0.07],
                       intensity=[0.15, 0.6], p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=[5, 17], p=0.3),
                A.MedianBlur(blur_limit=[5, 9], p=0.3),
            ]),
            
            
            A.Downscale(scale_min=0.4,
                        scale_max=0.9,
                        interpolation=cv2.INTER_LINEAR,
                        p=0.2),

            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(int(resize * 0.1), int(resize * 0.3)),
                hole_width_range=(int(resize * 0.1), int(resize * 0.3)),
                p=0.3
            ),

            A.ToGray(p=1.0),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD), 
            ToTensorV2()
        ])
    
    elif aug_type == "tl_occlusion_affine_domain_fda":
        
        ######
        # Import images from goal dataset
        import glob
        folder_path = "./active_learning/pool_unlabelled"
        limit=50

        image_paths = glob.glob(f"{folder_path}/*.jpg")
        if len(image_paths) > limit:
            image_paths = random.sample(image_paths, limit)
        
        refs = []
        for p in image_paths:
            img = cv2.imread(p)  
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (resize, resize))
            refs.append(img)

        ### Make transformations
        return A.Compose([
            A.Resize(width=resize, height=resize),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),

            A.Affine(
                scale=(0.90, 1.10),      # Zoom in/out by 80-120%
                rotate=(-180, 180),      # Rotate by -15 to +15 degrees
                translate_percent=(0, 0.05), # translate by 0-10%
                shear=(-2, 2),          # shear by -10 to +10 degrees
                p=0.8
            ),

            A.GridDistortion(num_steps=5,
                             distort_limit=[-0.30, 0.30]),
            
            A.Downscale(scale_min=0.4,
                        scale_max=0.9,
                        interpolation=cv2.INTER_LINEAR,
                        p=0.2),

            A.OneOf([
                A.HistogramMatching(
                    reference_images=refs,  # Utilise la variable 'refs' !
                    read_fn=identity_fn,    # Fonction simple
                    blend_ratio=(0.2, 0.5),
                    p=1.0
                ),
                A.FDA(
                    reference_images=refs,  # Utilise la variable 'refs' !
                    beta_limit=0.01,
                    read_fn=identity_fn,    # Fonction simple
                    p=1.0
                ),
            ], p=0.6),

            A.RandomGamma(gamma_limit=[40, 150], p=0.5),
            A.ISONoise(color_shift=[0.02, 0.07],
                       intensity=[0.15, 0.6], p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=[5, 17], p=0.3),
                A.MedianBlur(blur_limit=[5, 9], p=0.3),
            ]),
            
            
            A.Downscale(scale_min=0.5,
                        scale_max=0.9,
                        interpolation=cv2.INTER_LINEAR,
                        p=0.1),

            A.CoarseDropout(
                num_holes_range=(1, 3),
                hole_height_range=(int(resize * 0.1), int(resize * 0.3)),
                hole_width_range=(int(resize * 0.1), int(resize * 0.3)),
                p=0.3
            ),

            A.ToGray(p=1.0),
            A.Normalize(mean=NORM_MEAN, std=NORM_STD), 
            ToTensorV2()
        ])


def get_valid_transforms(data_config):
    resize = data_config["resize"]

    return A.Compose([
        A.Resize(width=resize, height=resize),
        A.Normalize(mean=NORM_MEAN, std=NORM_STD),
        A.ToGray(p=1.0),
        ToTensorV2()
    ])

def collect_samples_from_dir(root_dir):
    img_dir = os.path.join(root_dir, "imgs")
    mask_dir = os.path.join(root_dir, "masks") # Ou "masks" selon ton arborescence active learning
    
    # Support pour le dossier Active Learning qui s'appelle souvent pool_labelled/images et pool_labelled/masks
    if not os.path.exists(img_dir):
        img_dir = os.path.join(root_dir, "images")
    
    if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
        logging.warning(f"Ignored folder (couldn't find imgs/masks) : {root_dir}")
        return []

    img_names = sorted(os.listdir(img_dir))
    samples = []

    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        
        base_name = os.path.splitext(img_name)[0]
        possible_names = [img_name, base_name + ".png", base_name + ".jpg"]
        
        mask_path = None
        for pname in possible_names:
            p = os.path.join(mask_dir, pname)
            if os.path.exists(p):
                mask_path = p
                break
        
        if mask_path:
            samples.append((img_path, mask_path))
        else:
            # logging.debug(f"Masque manquant pour {img_name}")
            pass
            
    return samples


def get_dataloaders(data_config, use_cuda):
    valid_ratio = data_config["valid_ratio"]
    batch_size = data_config["batch_size"]
    num_workers = data_config["num_workers"]
    resize = data_config.get("resize", 224)
    training_type = data_config.get("training_type", "multisource")

    if training_type == "multisource":
        logging.info("  - Dataset creation")

        train_ds_full = PlanktonMultisourceDataset(
            sources_config=data_config["sources"],
            resize=resize,
            is_validation=False 
        )

        valid_ds_full = PlanktonMultisourceDataset(
            sources_config=data_config["sources"],
            resize=resize,
            is_validation=True 
        )

        logging.info(f"  - I loaded {len(train_ds_full)} samples.")

        original_indices = [i for i, x in enumerate(train_ds_full.samples) if x[3] == "original_dataset"]
        other_indices = [i for i, x in enumerate(train_ds_full.samples) if x[3] != "original_dataset"]

        logging.info(f"  - Split strategy: keeping validation strictly on 'original_dataset'.")
        logging.info(f"    Original samples: {len(original_indices)} | External samples (Pelgas, etc): {len(other_indices)}")

        original_labels = [train_ds_full.samples[i][1] for i in original_indices]

        if valid_ratio == 0:
            # Phase 3: use 100% of the data for training, no validation
            logging.info("  - valid_ratio=0: using all data for training (no validation split).")
            train_indices = original_indices + other_indices
            valid_indices = []
        else:
            # startified split on original dataset only
            train_idx_strat, valid_idx_strat = train_test_split(
                original_indices,
                test_size=valid_ratio,
                stratify=original_labels,
                random_state=42,
                shuffle=True
            )

            indices = list(range(len(train_ds_full)))
            random.seed(42)
            random.shuffle(original_indices)

            valid_indices = valid_idx_strat
            train_indices = train_idx_strat + other_indices

        train_dataset = torch.utils.data.Subset(train_ds_full, train_indices)
        valid_dataset = torch.utils.data.Subset(valid_ds_full, valid_indices) if valid_indices else None

        # Weighted Sampler
        logging.info("  - Computing class weights for Balanced Sampling...")
        
        # Compute the amount of images by class in the training subset
        train_labels = [train_ds_full.samples[i][1] for i in train_indices]
        class_counts = np.bincount(train_labels)
        base_class_weights = 1. / (class_counts + 1e-6)

        BOOST_FACTOR = 5.0 # We want to see 5x more the images from the original dataset

        sample_weights = []
        for i in train_indices:
            sample_entry = train_ds_full.samples[i]
            label = sample_entry[1]
            source = sample_entry[3]

            # Poids de base de la classe
            w = base_class_weights[label]

            if source == "original_dataset":
                w *= BOOST_FACTOR
            
            sample_weights.append(w)

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


        # Create the sampler
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True # Allows to retake the same image multiple times in one batch
        )


        # Build the dataloaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False, # Disabled because we use a sampler
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )

        if valid_dataset is not None:
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False, # No shuffle for validation, neither a sampler
                num_workers=num_workers,
                pin_memory=use_cuda,
            )
        else:
            valid_loader = None

        num_classes = len(train_ds_full.classes)
        input_size = tuple(train_ds_full[0][0].shape)

        return train_loader, valid_loader, input_size, num_classes
    
    elif training_type == "multicanal":
        logging.info("  - Dataset creation (Attention, pour le moment utilisation de transformations fixées)")

        # 1. Define Transforms (Torchvision)
        # We need specific transforms here because the input is a PIL Image (3 channels)
        # Note: for multicanal (Orig+CLAHE+Mask), we do NOT apply ToGray or
        # ImageNet normalization — each channel has its own semantic meaning.
        train_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                shear=5,
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 1.0))
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        valid_transform = transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        # 2. Instantiate Datasets
        # We create two instances pointing to the same data but with different transforms
        train_ds_full = PlanktonMultiChannelDataset(
            root_original=data_config["original_dataset"],
            root_clahe=data_config["clahe_dataset"],
            root_masks=data_config["masks_dataset"],
            transform=train_transform,
            is_test=False
        )

        valid_ds_full = PlanktonMultiChannelDataset(
            root_original=data_config["original_dataset"],
            root_clahe=data_config["clahe_dataset"],
            root_masks=data_config["masks_dataset"],
            transform=valid_transform, # Validation transform here
            is_test=False
        )

        logging.info(f"  - I loaded {len(train_ds_full)} samples available for Multi-Channel.")

        # 3. Stratified Split Logic
        # Since the dataset scans folders, we extract targets from paths to avoid loading all images
        targets = []
        for path in train_ds_full.image_paths:
            parent_dir = os.path.dirname(path).split(os.sep)[-1]
            targets.append(train_ds_full.class_to_idx.get(parent_dir, -1))

        if valid_ratio == 0:
            # Phase 3: use 100% of the data for training, no validation
            logging.info("  - valid_ratio=0: using all data for training (no validation split).")
            train_idx = np.arange(len(targets))
            valid_idx = np.array([], dtype=int)
        else:
            train_idx, valid_idx = train_test_split(
                np.arange(len(targets)),
                test_size=valid_ratio,
                stratify=targets,
                random_state=42,
                shuffle=True
            )

        # 4. Create Subsets
        train_dataset = torch.utils.data.Subset(train_ds_full, train_idx)
        valid_dataset = torch.utils.data.Subset(valid_ds_full, valid_idx) if len(valid_idx) > 0 else None

        logging.info(f"  - Split: {len(train_dataset)} train samples, {len(valid_dataset) if valid_dataset else 0} validation samples.")

        # --- weighted sampler ---
        train_targets = [targets[i] for i in train_idx]

        class_counts = np.bincount(train_targets)
        class_weights = 1. / (class_counts + 1e-6)
        sample_weights = [class_weights[t] for t in train_targets]

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

        # 5. Create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False, # Standard shuffle (add WeightedRandomSampler here if needed later)
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=use_cuda,
        )

        if valid_dataset is not None:
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=use_cuda,
            )
        else:
            valid_loader = None

        # Input is (Channels, H, W) -> (3, resize, resize)
        input_size = (3, resize, resize)
        num_classes = len(train_ds_full.classes)

        return train_loader, valid_loader, input_size, num_classes

        


def run_dataset_verification(sources_config, output_dir="data_output_test"):
    """
    Génère un rapport visuel complet :
    1. Grille Train (Augmentée)
    2. Grille Valid (Clean)
    3. Graphique de distribution
    4. Comparatif par classe (Original vs External) pour vérifier la cohérence.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import shutil
    import random
    from PIL import Image

    # 1. Préparation
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("DataVerifier")
    logger.info(f"=== Lancement de la vérification dans '{output_dir}' ===")

    # Config temporaire
    data_config = {
        "sources": sources_config,
        "valid_ratio": 0.2,
        "batch_size": 16, 
        "num_workers": 0, 
        "resize": 224
    }

    # 2. Création des Dataloaders
    try:
        # On force use_cuda=False pour le debug
        train_loader, valid_loader, input_size, num_classes = get_dataloaders(data_config, use_cuda=False)
    except Exception as e:
        logger.error(f"Erreur critique lors de la création des dataloaders : {e}")
        return

    # On récupère le dataset TRAIN complet (celui qui a les augmentations activées)
    full_dataset = train_loader.dataset.dataset 
    class_names = full_dataset.classes

    # --- UTILITAIRES ---
    def unnormalize(tensor):
        """Annule la normalisation ImageNet pour l'affichage"""
        mean = np.array(NORM_MEAN)
        std = np.array(NORM_STD)
        img = tensor.numpy().transpose((1, 2, 0)) # C, H, W -> H, W, C
        img = std * img + mean
        img = np.clip(img, 0, 1)
        return img

    def save_batch_grid(loader, title, filename, is_train):
        try:
            images, labels = next(iter(loader))
        except StopIteration:
            return

        aug_text = "Augmentations visibles" if is_train else "Pas d'augmentation"
        plt.figure(figsize=(12, 6))
        plt.suptitle(f"{title}\n({aug_text})", fontsize=14)

        count = min(len(images), 8)
        for i in range(count):
            ax = plt.subplot(2, 4, i + 1)
            img = unnormalize(images[i])
            label_name = class_names[labels[i]]
            plt.imshow(img)
            plt.title(f"{label_name}", fontsize=10)
            plt.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    # 3. Génération des visuels globaux
    save_batch_grid(train_loader, "TRAIN BATCH", "1_visualization_train.png", is_train=True)
    save_batch_grid(valid_loader, "VALID BATCH", "2_visualization_valid.png", is_train=False)

    # 4. Organisation des données pour analyse
    logger.info("Analyse de la distribution et cohérence...")
    original_root = sources_config["original_dataset"]["path"]
    
    # Structure : data_map[source][class_name] = [list_of_indices]
    data_map = {}
    sources_found = set()

    # On parcourt le dataset. 
    # ATTENTION : full_dataset.samples contient maintenant (path, label, transform, source)
    for idx, sample in enumerate(full_dataset.samples):
        # Déballage robuste (compatible 3 ou 4 éléments selon ta version)
        if len(sample) == 4:
            path, label_idx, transform_func, source_name = sample
        else:
            path, label_idx, transform_func = sample
            # Fallback si l'étape précédente n'est pas encore appliquée
            source_name = "original_dataset" if path.startswith(original_root) else "external"

        c_name = class_names[label_idx]
        sources_found.add(source_name)

        if source_name not in data_map:
            data_map[source_name] = {}
        if c_name not in data_map[source_name]:
            data_map[source_name][c_name] = []
        
        data_map[source_name][c_name].append(idx)

    # 5. Graphique de distribution
    classes_sorted = sorted(class_names)
    x = np.arange(len(classes_sorted))

    plt.figure(figsize=(24, 12))

    # On récupère les comptes pour l'original
    y_orig = [len(data_map.get("original_dataset", {}).get(c, [])) for c in classes_sorted]

    # On plot l'original en bas (VERT FIXE)
    p1 = plt.bar(x, y_orig, color='#2ca02c', label='Original Dataset', alpha=0.8)
    bottom_y = np.array(y_orig)

    # Palette de couleurs distinctes pour les ajouts (Rouge, Bleu, Orange, Violet, Cyan)
    # Cela garantit qu'on ne réutilise pas le vert par accident
    ext_colors = ['#d62728', '#1f77b4', '#ff7f0e', '#9467bd', '#17becf']
    color_idx = 0

    # On empile les autres sources
    for source in sources_found:
        if source == "original_dataset": continue

        y_src = [len(data_map.get(source, {}).get(c, [])) for c in classes_sorted]
        
        # Sélection de la couleur dans la liste
        c_code = ext_colors[color_idx % len(ext_colors)]
        color_idx += 1

        plt.bar(x, y_src, bottom=bottom_y, label=f'Added: {source}', color=c_code, alpha=0.8)
        bottom_y += np.array(y_src)

    plt.ylabel('Nombre d\'images')
    plt.title('Distribution des classes par Source', fontsize=18)
    plt.xticks(x, classes_sorted, rotation=90, fontsize=9)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_distribution_analysis.png"))
    plt.close()

    # 6. Vérification de la COHÉRENCE par classe (Side-by-Side)
    logger.info("Génération des planches de cohérence (Original vs External)...")

    # Pour chaque source externe ajoutée
    for source in sources_found:
        if source == "original_dataset": continue
        
        logger.info(f" -> Création planche pour : {source}")
        
        # Trouver les classes communes
        common_classes = [c for c in classes_sorted 
                          if c in data_map["original_dataset"] and c in data_map[source]]
        
        if not common_classes:
            logger.warning(f"Aucune classe commune entre original et {source}")
            continue

        # Création d'une image géante : 1 ligne par classe
        # Colonnes : 3 Original | 3 External
        n_rows = len(common_classes)
        fig, axes = plt.subplots(n_rows, 6, figsize=(18, 3 * n_rows))
        
        # Titre global
        fig.suptitle(f"Cohérence : Original (Gauche) vs {source} (Droite)\n(Avec Augmentations Train)", fontsize=20, y=1.005)

        # Si une seule classe, axes n'est pas un tableau 2D, on corrige
        if n_rows == 1: axes = np.array([axes])

        for row_idx, cls_name in enumerate(common_classes):
            # Indices disponibles
            idxs_orig = data_map["original_dataset"][cls_name]
            idxs_ext = data_map[source][cls_name]

            # Sélection aléatoire de 3 images max
            sel_orig = random.sample(idxs_orig, min(3, len(idxs_orig)))
            sel_ext = random.sample(idxs_ext, min(3, len(idxs_ext)))

            # Remplissage pour avoir toujours 3 items (répétition si besoin ou vide)
            while len(sel_orig) < 3 and len(sel_orig) > 0: sel_orig.append(sel_orig[0])
            while len(sel_ext) < 3 and len(sel_ext) > 0: sel_ext.append(sel_ext[0])

            # Affichage
            all_samples_row = sel_orig + sel_ext # [Orig, Orig, Orig, Ext, Ext, Ext]
            
            for col_idx, sample_idx in enumerate(all_samples_row):
                ax = axes[row_idx, col_idx]
                
                # Récupération et transformation
                path, _, transform_func, _ = full_dataset.samples[sample_idx]
                
                try:
                    img = Image.open(path).convert("RGB")
                    # On applique la transfo d'augmentation stockée
                    if transform_func:
                        img_tensor = transform_func(img)
                        img_disp = unnormalize(img_tensor)
                    else:
                        img_disp = np.array(img)
                    
                    ax.imshow(img_disp)
                except Exception as e:
                    ax.text(0.5, 0.5, "Error", ha='center')
                
                # Cosmétique
                ax.axis("off")
                
                # Titres de colonnes (seulement première ligne)
                if row_idx == 0:
                    if col_idx == 1: ax.set_title("ORIGINAL DATASET", fontsize=16, color='green', fontweight='bold')
                    if col_idx == 4: ax.set_title(f"SOURCE: {source}", fontsize=16, color='red', fontweight='bold')

            # Label de la classe à gauche de la ligne
            axes[row_idx, 0].text(-0.5, 0.5, cls_name, transform=axes[row_idx, 0].transAxes, 
                                  fontsize=14, va='center', ha='right', fontweight='bold')

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"4_coherence_{source}.jpg") # JPG pour compresser si c'est gros
        plt.savefig(save_path, dpi=100) # DPI modéré pour éviter fichier énorme
        plt.close()
        logger.info(f"Saved: {save_path}")

    logger.info("=== Vérification terminée ===")



if __name__ == "__main__":
    my_sources_config = {
        "original_dataset": {
                "path": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/data/original_dataset_optimized_512px/train",
                "augmentations_version": "basic",
                "classes_weights_file": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/classes_weights/original_dataset_max20000.jsonc"
            },
        "pelgas_dataset": {
                "path": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/data/pelgas_dataset_cropped_optimized_512px/101141/individual_images/",
                "augmentations_version": "pelgas_basic",
                "classes_weights_file": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/classes_weights/pelgas_dataset_0.jsonc",
                "classes_mapping_file": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/datasets_mappers/pelgas_to_original.jsonc"
            },
        "whoi_dataset": {
                "path": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/data/whoi_dataset_optimized_512px/",
                "augmentations_version": "pelgas_basic",
                "classes_weights_file": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/classes_weights/whoi_dataset_0.jsonc",
                "classes_mapping_file": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/datasets_mappers/whoi_to_original.jsonc"
            },
        "ndsb_dataset": {
                "path": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/data/ndsb_dataset_optimized_512px/train",
                "augmentations_version": "pelgas_basic",
                "classes_weights_file": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/classes_weights/ndsb_dataset_0.jsonc",
                "classes_mapping_file": "/home/lmalerba/projects/kaggle_challenge_sdi/kaggle_challenge_sdi/datasets_mappers/ndsb_to_original.jsonc"
            }
    }

    run_dataset_verification(my_sources_config, output_dir="data_output_test_original_max20000_full_pelgas_augmented")
