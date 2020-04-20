import math
import random

import cv2
import numpy as np
import pyvips
import torch
import csv
import torch.utils.data
import pathlib

# Save memory
pyvips.cache_set_max(0)

# Things can get messy with big images and multi-GPU:
cv2.setNumThreads(0)


class TissueDataset(torch.utils.data.Dataset):
    def __init__(self, img_size, img_dir, csv_fname, validation=False, augmentations=True, testing=False, 
                 limit_size=-1, variable_input_shapes=False, tile_size=4096, resize=1):
        self.img_dir = img_dir
        self.img_size = img_size
        self.validation = validation
        self.augmentations = augmentations
        self.variable_input_shapes = variable_input_shapes
        self.tile_size = tile_size
        self.tile_delta = tile_size
        self.testing = testing
        self.resize = resize

        images = []
        with open(csv_fname) as csvfile:
            csv_contents = csv.reader(csvfile, delimiter=',')
            for row in csv_contents: 
                fname = row[0]
                label = row[1]
                images.append((fname, label))
            images.pop(0)
        self.images = images

        assert len(images) > 0

    def __getitem__(self, index):
        img_fname, label = self.biopsy_fname_for_index(index)
        img = self.open_and_resize(img_fname)
        data = self.transforms(img)
        del img
        return data, label

    def biopsy_fname_for_index(self, index):
        img_fname, label = self.images[index]
        img_fname = pathlib.Path(self.img_dir) / pathlib.Path(img_fname)
        img_fname = str(img_fname)
        label = torch.tensor(int(label), dtype=torch.long)
        return img_fname, label

    def open_and_resize(self, img_fname):
        image = pyvips.Image.new_from_file(img_fname)

        if self.resize != 1:
            image = image.resize(self.resize, kernel="cubic")

        return image

    def transforms(self, image, save=False):
        # which interpolation to use for each transform
        interp = pyvips.Interpolate.new('bicubic')

        if not self.validation and self.augmentations:
            image = self.random_rotate(image, interp)
            image = self.random_flip(image)
            image = self.limit_size(image)
            image = self.color_jitter(image)
            image = self.elastic_transform(image, interp)

        # padding
        image = self.pad_image(image)

        # normalize
        tensor = np.ndarray(buffer=image.cast(pyvips.BandFormat.UCHAR).write_to_memory(),
                            dtype=np.uint8,
                            shape=[image.height, image.width, 3])

        tensor = tensor.transpose(2, 0, 1)
        tensor = torch.from_numpy(tensor)

        return tensor

    def pad_image(self, image):
        if not self.variable_input_shapes:
            image = image.gravity("centre", self.img_size, self.img_size, background=[255, 255, 255])
        else:
            # to play nicely with cudnn.benchmark, keep shapes equal
            image = self.pad_to_tile_delta(image)
        return image

    def pad_to_tile_delta(self, image):
        if image.width <= self.tile_size: w = self.tile_size
        else: w = math.ceil(image.width / self.tile_delta) * self.tile_delta
        if image.height <= self.tile_size: h = self.tile_size
        else: h = math.ceil(image.height / self.tile_delta) * self.tile_delta

        w = min(self.img_size, w)
        h = min(self.img_size, h)

        image = image.gravity("centre", w, h, background=[255, 255, 255])
        return image

    def limit_size(self, image):
        image = image.gravity("centre", min(self.img_size, image.width), min(self.img_size, image.height), background=[255, 255, 255])
        return image

    def elastic_transform(self, image, interp):
        def rand_matrix(width, height, alpha, sigma, random_state=np.random.RandomState(42), grid_scale=16):
            # Originally from https://github.com/rwightman/tensorflow-litterbox
            """Elastic deformation of images as per [Simard2003].  """
            alpha //= grid_scale
            sigma //= grid_scale

            # Downscaling the random grid and then upsizing post filter
            # improves performance. Approx 3x for scale of 4, diminishing returns after.
            grid_shape = (width // grid_scale, height // grid_scale)
            blur_size = int(grid_scale * sigma) | 1

            rand_x = cv2.GaussianBlur((random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
                                      ksize=(blur_size, blur_size),
                                      sigmaX=sigma) * alpha
            rand_y = cv2.GaussianBlur((random_state.rand(*grid_shape) * 2 - 1).astype(np.float32),
                                      ksize=(blur_size, blur_size),
                                      sigmaX=sigma) * alpha

            return rand_x, rand_y

        rand_x, rand_y = rand_matrix(self.img_size, self.img_size, alpha=1000, sigma=150, grid_scale=16)

        rand_x = rand_x[:image.height // 16, :image.width // 16]
        rand_y = rand_y[:image.height // 16, :image.width // 16]

        grid_scale = 16
        rand_x[0:grid_scale] = 0  # upper border
        rand_x[-grid_scale:] = 0  # lower border
        rand_x[:, 0:grid_scale] = 0  # left border
        rand_x[:, -grid_scale:] = 0  # right border

        rand_y[0:grid_scale] = 0
        rand_y[-grid_scale:] = 0
        rand_y[:, -grid_scale:] = 0
        rand_y[:, 0:grid_scale] = 0

        shift_map = np.stack([rand_x, rand_y], axis=2).flatten()
        trans_map = pyvips.Image.new_from_memory(shift_map.data, image.width // 16, image.height // 16, 2, 'float')

        coord_map = pyvips.Image.xyz(image.width // 16, image.height // 16)
        coord_map += trans_map
        coord_map *= 16
        image = image.mapim(coord_map.resize(16, kernel='linear'), interpolate=interp)
        return image

    def color_jitter(self, image):
        image = image.colourspace("lch")
        luminance_diff = random.uniform(0.95, 1.05)  # roughly brightness
        chroma_diff = random.uniform(0.95, 1.05)  # roughly saturation
        hue_diff = random.uniform(0.95, 1.05)  # hue
        image *= [luminance_diff, chroma_diff, hue_diff]
        image = image.colourspace("srgb")
        return image

    def random_flip(self, image):
        if random.choice([True, False]): image = image.fliphor()
        if random.choice([True, False]): image = image.flipver()
        return image

    def random_rotate(self, image, interp):
        random_angle = random.randint(0, 359)
        image = image.similarity(angle=random_angle, background=255, interpolate=interp)
        return image

    def __len__(self):
        return len(self.images)
