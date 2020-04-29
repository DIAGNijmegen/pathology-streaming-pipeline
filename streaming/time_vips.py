import pyvips
import numpy as np
import random
import torch
import cv2
import numpy as np

# from albumentations import (
#     RandomRotate90,
#     HueSaturationValue,
#     ElasticTransform,
#     Compose,
# )

# def strong_aug(p=0.5):
#     return Compose([
#         RandomRotate90(),
#         HueSaturationValue(p=1),
#         ElasticTransform(p=1, approximate=True)
#     ], p=p)


# pyvips.cache_set_max_mem(1000)

def transforms(image, save=False):
    image = pyvips.Image.new_from_file(image, access='sequential')

    # which interpolation to use for each transform
    interp = pyvips.Interpolate.new('bilinear')

    # image = limit_size(image)
    print(image.height, image.width)
    image = limit_size(image)
    image = random_rotate(image, interp)
    image = random_flip(image)
    image = color_jitter(image)
    image = limit_size(image)
    image = elastic_transform(image, interp)

    # padding
    image = pad_image(image)
    # return image

    # normalize
    tensor = np.ndarray(buffer=image.cast(pyvips.BandFormat.UCHAR).write_to_memory(),
                        dtype=np.uint8,
                        shape=[image.height, image.width, 3])

    # augmentation = strong_aug(p=0.9)
    # data = {"image": tensor}
    # augmented = augmentation(**data)

    tensor = tensor.transpose(2, 0, 1)
    tensor = torch.from_numpy(tensor)

    # return augmented
    return tensor

def pad_image(image):
    image = image.gravity("centre", 16384, 16384, background=[255, 255, 255])
    return image

def limit_size( image):
    image = image.gravity("centre", min(16384, image.width), min(16384, image.height), background=[255, 255, 255])
    return image

def elastic_transform(image, interp):
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

    rand_x, rand_y = rand_matrix(16384, 16384, alpha=1000, sigma=150, grid_scale=16)

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

def color_jitter(image):
    image = image.colourspace("lch")
    luminance_diff = random.uniform(0.95, 1.05)  # roughly brightness
    chroma_diff = random.uniform(0.95, 1.05)  # roughly saturation
    hue_diff = random.uniform(0.95, 1.05)  # hue
    image *= [luminance_diff, chroma_diff, hue_diff]
    image = image.colourspace("srgb")
    return image

def random_flip(image):
    if random.choice([True, False]): image = image.fliphor()
    if random.choice([True, False]): image = image.flipver()
    return image

def random_rotate(image, interp):
    random_angle = random.randint(0, 359)
    image = image.similarity(angle=random_angle, background=255, interpolate=interp)
    return image

if __name__ == "__main__":
    # image = transforms('/home/user/BM_S03_P000030_C0001_L01_A15.v')
    # image = transforms('/home/user/BM_S03_P000030_C0001_L01_A15.v')

    # image = pyvips.Image.new_from_file('/home/user/data/BM_S03_P000030_C0001_L01_A15.jpg')
    # image.write_to_file('/home/user/BM_S03_P000030_C0001_L01_A15.v')

    # image = pyvips.Image.new_from_file('/home/user/data/BM_S03_P000031_C0001_L01_A15.jpg')
    # image.write_to_file('/home/user/BM_S03_P000031_C0001_L01_A15.v')

    # image = pyvips.Image.new_from_file('/home/user/data/BM_S03_P000032_C0001_L01_A15.jpg')
    # image.write_to_file('/home/user/BM_S03_P000032_C0001_L01_A15.v')

    # filename = '/home/user/BM_S03_P000030_C0001_L01_A15.v'
    filename = '/home/user/data/BM_S03_P000030_C0001_L01_A15.jpg'
    from time import time
    start = time()
    image = transforms(filename)
    filename = '/home/user/BM_S03_P000031_C0001_L01_A15.v'
    print(time() - start, 's')
    start = time()
    image = transforms(filename)
    print(time() - start, 's')
    filename = '/home/user/BM_S03_P000032_C0001_L01_A15.v'
    start = time()
    image = transforms(filename)
    print(time() - start, 's')

    # Percent of CPU this job got: 2001%
    # 8.96011757850647 s
    # 8.779637336730957 s
    # 9.39746379852295 s
    # Maximum resident set size (kbytes): 3423144

    # image = transforms('/home/user/data/BM_S03_P000030_C0001_L01_A15.jpg')
    # image.write_to_file('/home/user/BM_S03_P000030_C0001_L01_A15.png')
