import pathlib

import cv2
import numpy as np
from joblib import Parallel, delayed
import subprocess
import math

import sys
from trim_tissue import TissueExtractor, create_parser

cv2.setNumThreads(0)

def count_docker_cpus(quota_file=None, period_file=None, log_fct=print):
    try:
        with open(quota_file or "/sys/fs/cgroup/cpu/cpu.cfs_quota_us", 'r') as content_file:
            cfs_quota_us = int(content_file.read())
        with open(period_file or "/sys/fs/cgroup/cpu/cpu.cfs_period_us", 'r') as content_file:
            cfs_period_us = int(content_file.read())
        if cfs_quota_us > 0 and cfs_period_us > 0:
            n_cpus = int(math.ceil(cfs_quota_us / cfs_period_us))
            # print('%d cpus accoriding to quota' % n_cpus)
            return n_cpus

    except Exception:
        log_fct("Getting number of cpus from cfs_quota failed; using multiprocessing.cpu_count", sys.exc_info())
    return 0

def create_parser_remote():
    parser = create_parser()
    parser.add_argument('--remote-dir', default=None, type=str, required=True)
    parser.set_defaults(remote_dir=-1)
    return parser

class TissueExtractorRemote(TissueExtractor):
    remote_dir: str = ''

    def __init__(self, remote_dir='', masks_dir='', mask_suffix='', output_spacing=1.0, threshold_downsample=32):
        super().__init__('/home/user/temp', masks_dir, mask_suffix, output_spacing, threshold_downsample)
        subprocess.check_output(["mkdir", "-p", '/home/user/temp'])
        self.remote_dir = remote_dir

    def save_sampling_mask(self, mask, fname):
        mask_fname = pathlib.Path(fname.parent, fname.stem + '_msk')
        np.save(str(mask_fname), mask)
        self.rsync_file(mask_fname.with_suffix('.npy'))

    def write_img(self, image, fname, quality=0):
        super().write_img(image, fname, quality)
        self.rsync_file(fname)

    def rsync_file(self, fname):
        try:
            subprocess.check_output(["rsync", "-vru", fname, self.remote_dir])
        except Exception as e:
            print('rsync failed with error:', e)

    def final_sync(self, fname):
        try:
            rsync_output = subprocess.check_output(["rsync", "-vru", self.save_dir, self.remote_dir])
            print(rsync_output)
        except Exception as e:
            print('rsync failed with error:', e)

def extract_tissue(file, remote_dir, masks_dir, masks_suffix, output_spacing):
    extractor = TissueExtractorRemote(remote_dir, masks_dir, masks_suffix, output_spacing)

    fname = pathlib.Path(file)
    save_dir = pathlib.Path('/home/user/temp')
    new_fname = save_dir / fname.stem
    new_fname = new_fname.with_suffix('.jpg')

    extractor.extract_file(fname, new_fname)


if __name__ == '__main__':
    parser = create_parser_remote()
    args = parser.parse_args()
    args.remote_dir += '/'

    print(args)

    num_cores = count_docker_cpus()
    if num_cores == 0 or args.num_processes > -1:
        num_cores = args.num_processes
    else:
        print("Using number of cpu's from docker:", num_cores)

    if args.image:
        extract_tissue(args.image, args.remote_dir, args.masks_dir, args.mask_suffix, args.output_spacing)
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
                                                           args.remote_dir,
                                                           args.masks_dir,
                                                           args.mask_suffix,
                                                           args.output_spacing) for file in images)

    print()
    print('Final sync - should not transfer any more files (otherwise chansey missed some..)')
    missed = True
    retry_counter = 0
    while missed and retry_counter < 10:
        try:
            rsync_output = subprocess.check_output(["rsync", "-ru", "--stats", '/home/user/temp/', args.remote_dir[:-1]])
            rsync_output = rsync_output.decode("utf-8")
            if [int(s) for s in rsync_output.split('\n')[4].split() if s.isdigit()][0] > 0: 
                retry_counter += 1
                print('Still transferred files, try again')
                print(rsync_output)
            else:
                missed = False
                print()
                print('Everything is successfully transferred')

        except Exception as e:
            print('rsync failed with error:', e)
            retry_counter += 1
