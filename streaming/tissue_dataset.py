import math
import random

import cv2
import numpy as np
import pyvips
import torch
import csv
import torch.utils.data
import pathlib
import os

# Save memory, seems to make it a wee bit slower.
# pyvips.cache_set_max(0)

# Things can get messy with big images and multi-GPU:
cv2.setNumThreads(0)


class TissueDataset(torch.utils.data.Dataset):
    def __init__(self, img_size, img_dir, filetype, csv_fname, validation=False, augmentations=True, 
                 limit_size=-1, variable_input_shapes=False, tile_size=4096, resize=1, multiply_len=1, num_classes=2,
                 regression=False):
        self.img_dir = img_dir
        self.img_size = img_size
        self.filetype = filetype
        self.validation = validation
        self.augmentations = augmentations
        self.variable_input_shapes = variable_input_shapes
        self.tile_size = tile_size
        self.tile_delta = tile_size
        self.resize = resize
        self.multiply_len = multiply_len
        self.num_classes = num_classes
        self.regression = regression

        images = []
        with open(csv_fname) as csvfile:
            csv_contents = csv.reader(csvfile, delimiter=',')
            for row in csv_contents: 
                fname = row[0]
                label = row[1]
                images.append((fname, label))
            images.pop(0)
        self.images = images

        included = []
        for i in range(len(self)):
            fname = self.biopsy_fname_for_index(i)[0]
            if os.path.isfile(fname):
                included.append(self.images[i])
            else:
                print('WARNING', fname, 'not found, excluded!')

        self.images = included
        assert len(self.images) > 0

    def __getitem__(self, index):
        index = index // self.multiply_len
        img_fname, mask_fname, label = self.biopsy_fname_for_index(index)
        img = self.open_and_resize(img_fname)
        if os.path.isfile(mask_fname):
            mask = np.load(mask_fname)
        else:
            mask = None
        data = self.transforms(img, mask)
        del img
        return data, label

    def biopsy_fname_for_index(self, index):
        img_fname, label = self.images[index]
        img_path = pathlib.Path(self.img_dir) / pathlib.Path(img_fname).with_suffix(self.filetype)
        mask_fname = pathlib.Path(self.img_dir) / pathlib.Path(img_fname + '_msk')
        mask_path = mask_fname.with_suffix('.npy')
        if '[' in label: 
            label = torch.tensor(eval(label), dtype=torch.long)
            label = torch.nn.functional.one_hot(label, num_classes=self.num_classes).sum(dim=0).float()
        elif self.regression: label = torch.tensor(float(label), dtype=torch.float32)
        else: label = torch.tensor(int(label), dtype=torch.long)
        return str(img_path), str(mask_path), label

    def open_and_resize(self, img_fname):
        image = pyvips.Image.new_from_file(img_fname)

        if self.resize != 1:
            image = image.resize(self.resize, kernel="cubic")

        return image

    def transforms(self, image, mask=None, save=False):
        # which interpolation to use for each transform
        interp = pyvips.Interpolate.new('bilinear')

        if not self.validation and self.augmentations:
            image = self.random_flip(image)
            image = self.random_rotate(image, interp)
            image = self.limit_size(image, mask)
            image = self.color_jitter(image)
            image = self.elastic_transform(image, interp)
        elif mask is not None:
            # use highest value in mask for validation
            image = self.crop_using_mask(mask, image, random_index=False)

        # padding
        image = self.pad_image(image)

        # normalize
        tensor = np.ndarray(buffer=image.cast(pyvips.BandFormat.UCHAR).write_to_memory(),
                            dtype=np.uint8,
                            shape=[image.height, image.width, 3])

        tensor = tensor.transpose(2, 0, 1)
        tensor = torch.from_numpy(tensor)

        return tensor

    def pad_image(self, image, mask=None):
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

    def limit_size(self, image, mask=None):
        if mask is not None:
            image = self.crop_using_mask(mask, image)
        else:
            image = image.gravity("centre", min(self.img_size, image.width), min(self.img_size, image.height), background=[255, 255, 255])

        return image

    def crop_using_mask(self, mask, image, random_index=True):
        norm_mask = mask / np.sum(mask)
        if random_index:
            rand_i = np.random.choice(mask.size, p=norm_mask.flatten())
        else:
            rand_i = np.argmax(norm_mask.flatten())

        # 'reshape' index back to 2d-coordinates
        nrows = rand_i / mask.shape[0]
        y = math.floor(nrows)
        x = (rand_i - y*mask.shape[0])

        # mask is 100x smaller than image
        y, x = y * 100, x * 100  

        # index is center of tile
        y, x = y - self.img_size // 2, x - self.img_size // 2  

        if x + self.img_size > image.width:
            x = image.width - self.img_size - 1
        if y + self.img_size > image.height:
            y = image.height - self.img_size - 1

        if x < 0: x = 0
        if y < 0: y = 0

        image = image.crop(x, y, min(image.width, self.img_size), min(image.height, self.img_size))
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
        return len(self.images) * self.multiply_len
