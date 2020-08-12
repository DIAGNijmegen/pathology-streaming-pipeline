import math
import random

from time import time
import cv2
import numpy as np
import pyvips
import torch
import csv
import torch.utils.data
import pathlib
import os
import shutil

# Save memory, seems to make it a wee bit slower.
# pyvips.cache_set_max(0)

# Things can get messy with big images and multi-GPU:
cv2.setNumThreads(0)

class TissueDataset(torch.utils.data.Dataset):
    def __init__(self, img_size, img_dir, cache_dir, filetype, csv_fname, augmentations=True, 
                 limit_size=-1, variable_input_shapes=False, tile_size=4096, resize=1, multiply_len=1, num_classes=2,
                 regression=False, convert_to_vips=False):
        self.img_dir = img_dir
        self.cache_dir = cache_dir
        self.img_size = img_size
        self.filetype = filetype
        self.augmentations = augmentations
        self.variable_input_shapes = variable_input_shapes
        self.tile_size = tile_size
        self.tile_delta = tile_size
        self.resize = resize
        self.multiply_len = multiply_len
        self.num_classes = num_classes
        self.regression = regression
        self.convert_to_vips = convert_to_vips

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
        if limit_size > -1:
            self.images = self.images[0:limit_size]
            self.labels = [torch.randint(0, self.num_classes, size=(1,)).item() for _ in range(limit_size)]
        assert len(self.images) > 0

    def __getitem__(self, index):
        self.index = index
        try:
            data, label = self.get_data_label_for_index(index)
            return data, label
        except pyvips.error.Error as e:
            print(e)
            cache_fname = self.biopsy_fname_for_index(self.index)[2]
            os.remove(cache_fname)
            # this could result in a loop
            return self.get_data_label_for_index(self.index)

    def get_data_label_for_index(self, index):
        index = index // self.multiply_len
        img_fname, mask_fname, cache_fname, label = self.biopsy_fname_for_index(index)
        img = self.open_and_resize(img_fname, cache_fname)
        if os.path.isfile(mask_fname): mask = np.load(mask_fname)
        else: mask = None
        data = self.transforms(img, mask)
        del img
        return data, label

    def biopsy_fname_for_index(self, index):
        img_fname, label = self.images[index]
        img_path = pathlib.Path(self.img_dir) / pathlib.Path(img_fname).with_suffix(self.filetype)
        stem = pathlib.Path(img_fname).stem
        if self.convert_to_vips:
            cache_path = pathlib.Path(self.cache_dir) / pathlib.Path(stem + '_cache').with_suffix('.v')
        else:
            cache_path = pathlib.Path(self.cache_dir) / pathlib.Path(stem + '_cache').with_suffix(self.filetype)
        mask_fname = pathlib.Path(self.img_dir) / pathlib.Path(img_fname + '_msk')
        mask_path = mask_fname.with_suffix('.npy')
        if '[' in label: 
            label = torch.tensor([int(i) for i in label.split(';')], dtype=torch.long)
            label = torch.nn.functional.one_hot(label, num_classes=self.num_classes).sum(dim=0).float()
        elif self.regression: label = torch.tensor(float(label), dtype=torch.float32)
        else: label = torch.tensor(int(label), dtype=torch.long)
        return str(img_path), str(mask_path), str(cache_path), label

    def open_and_resize(self, img_fname, cache_fname):
        if self.cache_dir and os.path.isfile(cache_fname):
            image = pyvips.Image.new_from_file(cache_fname, memory=False)
        else:
            if self.resize != 1:
                image = pyvips.Image.new_from_file(img_fname, shrink=1/self.resize)
                if self.cache_dir: image.write_to_file(cache_fname)
            else:
                image = pyvips.Image.new_from_file(img_fname)
                if self.cache_dir: shutil.copyfile(img_fname, cache_fname)
        return image

    def transforms(self, image, mask=None, save=False):
        # which interpolation to use for each transform
        interp = pyvips.Interpolate.new('bilinear')

        if self.augmentations:
            image = self.random_rotate(image, interp)
            image = self.random_flip(image)
            image = self.limit_size(image, mask)
            image = self.color_jitter(image)
            image = self.elastic_transform_old(image, interp)
        elif mask is not None and not self.variable_input_shapes:
            # use highest value in mask for validation
            image = self.crop_using_mask(mask, image, random_index=False)

        # padding
        image = self.pad_or_crop_image(image)

        # normalize
        tensor = np.ndarray(buffer=image.cast(pyvips.BandFormat.UCHAR).write_to_memory(),
                            dtype=np.uint8,
                            shape=[image.height, image.width, 3])

        tensor = tensor.transpose(2, 0, 1)
        tensor = torch.from_numpy(tensor)

        return tensor

    def pad_or_crop_image(self, image, mask=None):
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

        image = image.gravity("centre", w, h, background=[255, 255, 255])
        return image

    def limit_size(self, image, mask=None):
        if mask is not None:
            image = self.crop_using_mask(mask, image)
        else:
            image = self.random_crop(image)

        return image

    def random_crop(self, image):
        cx = np.random.randint(0, max(1, image.width - self.img_size))
        cy = np.random.randint(0, max(1, image.height - self.img_size))
        image = image.crop(cx, cy, min(image.width, self.img_size), min(image.height, self.img_size))
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

    def elastic_transform_old(self, image, interp, adjust_for_image_size=False):
        def rand_matrix(width, height, alpha, sigma, grid_scale=16):
            # Originally from https://github.com/rwightman/tensorflow-litterbox
            """Elastic deformation of images as per [Simard2003].  """
            alpha //= grid_scale
            sigma //= grid_scale

            # Downscaling the random grid and then upsizing post filter
            # improves performance. Approx 3x for scale of 4, diminishing returns after.
            grid_shape = (height // grid_scale, width // grid_scale)
            blur_size = int(grid_scale * sigma) | 1

            rand_x = cv2.GaussianBlur((np.random.rand(*grid_shape) * 2 - 1).astype(np.float32),
                                      ksize=(blur_size, blur_size),
                                      sigmaX=sigma) * alpha
            rand_y = cv2.GaussianBlur((np.random.rand(*grid_shape) * 2 - 1).astype(np.float32),
                                      ksize=(blur_size, blur_size),
                                      sigmaX=sigma) * alpha

            return rand_x, rand_y

        rand_x, rand_y = rand_matrix(image.width, image.height, alpha=1000, sigma=150, grid_scale=16)

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
        trans_map = trans_map.resize(16, kernel='linear')
        coord_map = pyvips.Image.xyz(image.width, image.height)
        coord_map += trans_map

        image = image.mapim(coord_map, interpolate=interp)

        return image

    def elastic_transform(self, image, interp, adjust_for_image_size=False):
        def rand_matrix(width, height, alpha, sigma, grid_scale=16):
            # Originally from https://github.com/rwightman/tensorflow-litterbox
            """Elastic deformation of images as per [Simard2003].  """
            # Downscaling the random grid and then upsizing post filter
            # improves performance. Approx 3x for scale of 4, diminishing returns after.
            grid_shape = (height, width)

            if sigma[0] % 2 == 0: sigma = (sigma[0]+1, sigma[1])
            if sigma[1] % 2 == 0: sigma = (sigma[0], sigma[1]+1)

            rand_x = cv2.GaussianBlur((np.random.rand(*grid_shape) * 2 - 1).astype(np.float32),
                                      ksize=(3*sigma[1], 3*sigma[1]),
                                      sigmaX=sigma[1]) * alpha[1]
            rand_y = cv2.GaussianBlur((np.random.rand(*grid_shape) * 2 - 1).astype(np.float32),
                                      ksize=(3*sigma[0], 3*sigma[0]),
                                      sigmaX=sigma[0]) * alpha[0]

            return rand_x, rand_y

        image = image.gravity("centre", image.width + 512, image.height + 512, background=[255, 255, 255])

        sigma_w, sigma_h = 32, 32
        alpha_h, alpha_w = 256, 256

        grid_scale = self.img_size // 256

        rand_x, rand_y = rand_matrix(image.width // grid_scale, image.height // grid_scale,
                                     alpha=(alpha_h, alpha_w),
                                     sigma=(sigma_h, sigma_w), grid_scale=grid_scale)

        rand_x = rand_x[:image.height // grid_scale, :image.width // grid_scale]
        rand_y = rand_y[:image.height // grid_scale, :image.width // grid_scale]

        border = grid_scale // 2
        rand_x[0:border] = 0  # upper border
        rand_x[-border:] = 0  # lower border
        rand_x[:, 0:border] = 0  # left border
        rand_x[:, -border:] = 0  # right border

        rand_y[0:border] = 0
        rand_y[-border:] = 0
        rand_y[:, -border:] = 0
        rand_y[:, 0:border] = 0

        shift_map = np.stack([rand_x, rand_y], axis=2).flatten()
        trans_map = pyvips.Image.new_from_memory(shift_map.data, image.width // grid_scale, image.height // grid_scale, 2, 'float')

        coord_map = pyvips.Image.xyz(image.width // grid_scale, image.height // grid_scale)
        coord_map += trans_map
        coord_map *= grid_scale
        image = image.mapim(coord_map.resize(grid_scale, kernel='cubic'), interpolate=interp)
        image = image.gravity("centre", image.width - 512, image.height - 512, background=[255, 255, 255])

        return image

    def color_jitter(self, image, delta=0.05):
        image = image.colourspace("lch")
        luminance_diff = random.uniform(1.0-delta, 1.0+delta)  # roughly brightness
        chroma_diff = random.uniform(1.0-delta, 1.0+delta)  # roughly saturation
        hue_diff = random.uniform(1.0-delta, 1.0+delta)  # hue
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
