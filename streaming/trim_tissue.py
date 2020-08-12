import argparse
import math
import pathlib

import cv2
import os
import numpy as np
from joblib import Parallel, delayed

import multiresolutionimageinterface as mir

cv2.setNumThreads(0)

def create_parser():
    parser = argparse.ArgumentParser(description='Trim existing tissue and save jpg')

    parser.add_argument('--slides-dir', default=None, type=str, required=False)
    parser.add_argument('--filetype', default='tif', type=str, required=False, help='mrxs, or tif etc')
    parser.add_argument('--save-dir', default=None, type=str, required=False)
    # parser.add_argument('--overwrite', action='store_false')

    parser.add_argument('--image', default=None, type=str)
    parser.add_argument('--csv', default=None, type=str, required=False)
    parser.add_argument('--csv-row', default=0, type=int, required=False)

    parser.add_argument('--masks-dir', default=None, type=str, required=False)
    parser.add_argument('--mask-suffix', default=None, type=str, required=False)
    # parser.add_argument('--mask-level', default=0, type=int, required=False)

    parser.add_argument('--output-spacing', default=1.0, type=float, required=False, help='downsample the wsi for sharpness')

    parser.add_argument('--num-processes', default=4, type=int, help='how many processes to parallely extract')
    parser.add_argument('--dont_remove_empty_regions', action='store_false')

    return parser

def zero_runs(a):
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges

def search_empty_regions(arr, axis, sum_axis, min_length_range=2):
    summed_axis = np.sum(arr, sum_axis)
    zero_columns_ranges = zero_runs(summed_axis)
    trim_to = min_length_range

    filtered_zero_ranges = []
    for start, end in zero_columns_ranges:
        length = end - start
        if length > min_length_range:
            start += trim_to // 2
            end -= trim_to // 2
            new_range = [start, end]
            filtered_zero_ranges.append(new_range)

    if len(filtered_zero_ranges) == 0:
        return None
    return np.vstack(filtered_zero_ranges)

def empty_regions_2d(arr, min_length_range=2):
    rows = search_empty_regions(arr, 0, 1, min_length_range)
    columns = search_empty_regions(arr, 1, 0, min_length_range)
    return (rows, columns)

class TissueExtractor(object):
    threshold_level: int
    threshold_downsample: int

    read_level: int = 1
    read_downsample:int = -1
    read_tile_size: int = 512
    write_downsample: int = 2

    mask_level: int
    masks_dir: pathlib.Path

    diff_mask_to_read_level: int
    diff_mask_to_level_0: int
    diff_read_to_level_0: int

    reader: mir.MultiResolutionImage
    mask: np.ndarray

    def __init__(self, save_dir: str, masks_dir: str, mask_suffix: str, output_spacing=1.0, threshold_downsample=32, remove_empty_regions=True):
        super().__init__()
        assert save_dir
        self.threshold_downsample = threshold_downsample
        self.save_dir = pathlib.Path(save_dir)
        self.output_spacing = output_spacing
        self.masks_dir = pathlib.Path(masks_dir) if masks_dir else None  # type:ignore
        self.mask_suffix = mask_suffix
        self.should_remove_empty_regions = remove_empty_regions

    def extract_files(self, files):
        for file in files:
            fname = pathlib.Path(file)
            new_fname = self.save_dir / fname.stem
            self.extract_file(fname, new_fname)

    def extract_file(self, file, fname, suffix='.jpg'):
        print(file)
        if os.path.isfile(fname.with_suffix(suffix)):
            print('Already exist', file, fname.with_suffix(suffix))
            return

        self.open_slide(file)
        self.calculate_read_level()
        self.create_mask(fname)

        coords = self.bounding_box()
        empty_rows, empty_cols = self.find_empty_regions(coords)

        sampling_mask = self.create_sampling_mask(self.mask, coords)
        empty_rows_mask = self.translate_regions_to_sampling_mask(empty_rows)
        empty_cols_mask = self.translate_regions_to_sampling_mask(empty_cols)
        if self.should_remove_empty_regions:
            sampling_mask = self.remove_empty_regions(sampling_mask, empty_rows_mask, empty_cols_mask)
        self.save_sampling_mask(sampling_mask, fname)

        image = self.fetch_region(coords)
        empty_rows_img = self.translate_regions_to_image(empty_rows)
        empty_cols_img = self.translate_regions_to_image(empty_cols)
        if self.should_remove_empty_regions:
            image = self.remove_empty_regions(image, empty_rows_img, empty_cols_img)
        self.write_img(image, fname.with_suffix(suffix))

        print('Wrote', fname.with_suffix(suffix))

    def create_mask(self, fname):
        maskfn = None
        if self.masks_dir:
            maskfn = str(self.masks_dir / fname.stem) + self.mask_suffix
            if os.path.isfile(maskfn):
                self.read_threshold_mask(maskfn)
                non_zero_mask = np.where(self.mask)
                if len(non_zero_mask[0]) == 0:
                    print(fname.stem, '! WARNING: mask is empty, generating based on thresholding tissue')
                    maskfn = None
            else:
                print(fname.stem, '! WARNING: mask file does not exist, generating based on thresholding tissue', maskfn)
                maskfn = None

        self.calculate_threshold_level()
        self.calculate_downsamples()

        if not maskfn:
            self.create_threshold_mask()

    def translate_regions_to_image(self, regions):
        if regions is None:
            return None
        regions = regions.copy()
        regions = regions.astype(np.float32)
        regions *= self.diff_mask_to_read_level
        regions /= self.write_downsample
        return np.round(regions).astype(np.uint32)

    def translate_regions_to_sampling_mask(self, regions):
        if regions is None:
            return None
        regions = regions.copy()
        regions = regions.astype(np.float32)
        regions *= self.diff_mask_to_read_level
        regions /= self.write_downsample
        regions /= 100
        return np.round(regions).astype(np.uint32)

    def calculate_read_level(self):
        downsample = int(round(self.output_spacing / self.reader.getSpacing()[0]))
        level = self.reader.getBestLevelForDownSample(downsample)
        level_downsample = self.reader.getLevelDownsample(level)
        while level_downsample > downsample:
            level -= 1
            level_downsample = self.reader.getLevelDownsample(level)
            if level == 0:
                print('Required output spacing not found, using level 0, write downsample', downsample / level_downsample, 'spacing', self.reader.getSpacing()[0])
                break

        self.read_level = level
        self.write_downsample = downsample / level_downsample

    def calculate_threshold_level(self):
        self.threshold_level = self.reader.getBestLevelForDownSample(self.threshold_downsample)

    def calculate_downsamples(self):
        self.threshold_downsample = int(self.reader.getLevelDownsample(self.threshold_level))
        self.read_downsample = int(self.reader.getLevelDownsample(self.read_level))
        self.diff_mask_to_read_level = int(round(self.threshold_downsample / self.read_downsample))
        self.diff_mask_to_level_0 = int(self.reader.getLevelDownsample(self.threshold_level))

    def calculate_mask_downsamples(self, mask):
        self.diff_mask_to_level_0 = (mask.getSpacing()[0] * mask.getLevelDownsample(self.threshold_level)) \
            / (self.reader.getSpacing()[0])
        self.diff_mask_to_read_level = (mask.getSpacing()[0] * mask.getLevelDownsample(self.threshold_level)) \
            / (self.reader.getSpacing()[0] * self.reader.getLevelDownsample(self.read_level))

    def open_slide(self, file):
        self.reader = mir.MultiResolutionImageReader().open(str(file))

    def create_threshold_mask(self):
        dimensions = self.reader.getLevelDimensions(self.threshold_level)
        wsi_high_level_mask = self.reader.getUCharPatch(0, 0, dimensions[0], dimensions[1], self.threshold_level)

        blur = cv2.blur(wsi_high_level_mask, (100, 100))
        blur = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)

        wsi_high_level_mask = (blur > 235)
        wsi_high_level_mask += (blur == 0)
        wsi_high_level_mask = (wsi_high_level_mask == 0).astype(np.uint8)

        self.mask = wsi_high_level_mask

    def create_sampling_mask(self, mask, coords, train_tile_size=16384):
        y, height, x, width = coords

        mask_height = int(height * (self.diff_mask_to_read_level / self.write_downsample) / 100)
        mask_width = int(width * (self.diff_mask_to_read_level / self.write_downsample) / 100)
        small = cv2.resize(mask[y:y+height, x:x+width], (mask_width, mask_height))

        kernel = np.ones((int(train_tile_size / 100 // self.write_downsample), 
                          int(train_tile_size / 100 // self.write_downsample)))

        #filter the source image
        sampling = cv2.filter2D(small.astype('float'), -1, kernel)
        sampling = sampling / sampling.max()  # normalize
        sampling = sampling * sampling
        return sampling

    def save_sampling_mask(self, mask, fname):
        mask_fname = pathlib.Path(fname.parent, fname.stem + '_msk')
        np.save(str(mask_fname), mask)

    def read_threshold_mask(self, mask_fn):
        mask = mir.MultiResolutionImageReader().open(mask_fn)  # type: mir.MultiResolutionImage

        read_level = 0
        whole_mask = mask.getLevelDimensions(read_level)
        wsi_high_level_mask = mask.getUCharPatch(0, 0, whole_mask[0], whole_mask[1], read_level)

        blur = cv2.blur(wsi_high_level_mask * 255, (300, 300))
        wsi_high_level_mask = (blur > 128)
        kernel = np.ones((128, 128), np.uint8)

        self.mask = cv2.dilate(wsi_high_level_mask.astype(np.uint8), kernel, iterations=1)

        self.threshold_downsample = int(round(mask.getSpacing()[0] * mask.getLevelDownsample(read_level) / self.reader.getSpacing()[0]))

    def bounding_box(self):
        non_zero_mask = np.where(self.mask)
        y = int(np.min(non_zero_mask[0]))
        height = int(np.max(non_zero_mask[0]) - y)
        x = int(np.min(non_zero_mask[1]))
        width = int(np.max(non_zero_mask[1]) - x)
        return y, height, x, width

    def find_empty_regions(self, coords):
        y, height, x, width = coords
        return empty_regions_2d(self.mask[y:y+height, x:x+width], min_length_range=64)

    def split_regions(self, empty_rows, empty_cols, coords):
        """ This function can split multiple cross-sections on a slide
        not being used"""
        width, height = self.reader.getLevelDimensions(self.threshold_level)

        if width > height:
            empty_region = empty_cols
            axis = width
            perc_non_zero = coords[3] / width
        else:
            empty_region = empty_rows
            axis = height
            perc_non_zero = coords[1] / height

        if perc_non_zero < 0.7:
            return coords, None

        min_len_separation = axis / 100 * 1  # 1%

        possible_section_bounds = []
        for i, reg in enumerate(empty_region):
            len_reg = reg[1] - reg[0]
            mid_reg = (reg[0] + reg[1]) / 2
            pos_reg = mid_reg / axis
            if len_reg > min_len_separation and pos_reg > 0.25 and pos_reg < 0.75:
                possible_section_bounds.append((pos_reg, i))

        if len(possible_section_bounds) > 0:
            possible_section_bounds = np.array(possible_section_bounds)
            dist_to_mid = possible_section_bounds[:,0] - 0.5
            mid_bound = np.abs(dist_to_mid).argmin()
            mid_bound = int(possible_section_bounds[mid_bound][1])
            bound = empty_region[mid_bound]
            y, height, x, width = coords
            if axis == width:
                first_section = (y, height, x, bound[0])
                second_section = (y, height, bound[1], width - bound[1])
            else:
                first_section = (y, bound[0], x, width)
                second_section = (int(bound[1]), height - bound[1], x, width)
            return first_section, second_section
        return coords, None

    def fetch_region(self, coords):
        y, height, x, width = coords
        tiles_size = self.read_tile_size

        n_rows = math.ceil(height * self.diff_mask_to_read_level / tiles_size)
        n_cols = math.ceil(width * self.diff_mask_to_read_level / tiles_size)

        resized_ts = int(tiles_size / self.write_downsample)
        placeholder = np.zeros((n_rows * resized_ts,
                                n_cols * resized_ts, 3), dtype=np.uint8)

        y *= int(round(self.diff_mask_to_level_0))
        x *= int(round(self.diff_mask_to_level_0))

        print("New size:", placeholder.shape[:-1])
        for i in range(n_rows):
            for j in range(n_cols):
                tile_y = y + tiles_size * i * self.read_downsample
                tile_x = x + tiles_size * j * self.read_downsample

                tile = self.reader.getUCharPatch(tile_x, tile_y, tiles_size, tiles_size, self.read_level)
                tile[np.sum(tile, axis=2) < 30] = 255  # invalid tile fix

                if tile.shape[0] < tiles_size or tile.shape[1] < tiles_size:
                    new_tile = np.ones((tiles_size, tiles_size, 3), dtype=np.uint8) * 255
                    new_tile[0:tile.shape[0], 0:tile.shape[1]] = tile
                    tile = new_tile

                if resized_ts != tiles_size:
                    tile = cv2.resize(tile, dsize=(resized_ts, resized_ts))

                ph_y = resized_ts * i
                ph_x = resized_ts * j
                placeholder[ph_y:ph_y + resized_ts,
                            ph_x:ph_x + resized_ts] = tile

        return placeholder

    def remove_empty_regions(self, image, empty_rows, empty_columns):
        if empty_rows is not None:
            empty_rows = np.hstack([[c for c in range(*r)] for r in empty_rows])
            image = np.delete(image, empty_rows, 0)

        if empty_columns is not None:
            empty_columns = np.hstack([[c for c in range(*r)] for r in empty_columns])
            image = np.delete(image, empty_columns, 1)

        return image

    def write_img(self, image, fname, quality=0):
        if quality == 0:
            cv2.imwrite(str(fname), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            cv2.imwrite(str(fname), cv2.cvtColor(image, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), quality])

def extract_tissue(file, save_dir, masks_dir, mask_suffix, output_spacing, remove_empty_regions):
    extractor = TissueExtractor(save_dir, masks_dir, mask_suffix, output_spacing, remove_empty_regions)

    fname = pathlib.Path(file)
    save_dir = pathlib.Path(save_dir)
    new_fname = save_dir / fname.stem
    new_fname = new_fname.with_suffix('.jpg')

    extractor.extract_file(fname, new_fname)


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    num_cores = args.num_processes

    if args.image:
        extract_tissue(args.image, args.save_dir, args.masks_dir, args.mask_suffix, args.output_spacing, args.remove_empty_regions)
    else:
        images_dir = pathlib.Path(args.slides_dir) if args.slides_dir else None
        if args.csv:
            import csv
            images = []
            with open(args.csv) as csvfile:
                readCSV = csv.reader(csvfile, delimiter=',')
                for row in readCSV: 
                    fname = pathlib.Path(row[args.csv_row])
                    images.append(images_dir / fname.with_suffix('.' + args.filetype))
                images.pop(0)
        else:
            images = list(images_dir.glob('*.' + args.filetype))

        Parallel(n_jobs=num_cores)(delayed(extract_tissue)(file,
                                                           args.save_dir,
                                                           args.masks_dir,
                                                           args.mask_suffix,
                                                           args.output_spacing, 
                                                           args.remove_empty_regions) for file in images)
