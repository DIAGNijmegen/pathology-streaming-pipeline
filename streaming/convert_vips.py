import pyvips
import glob
from tqdm import tqdm

if __name__ == "__main__":
    images = glob.glob('/home/user/data/*.jpg')
    for image_fname in tqdm(images):
        image = pyvips.Image.new_from_file(image_fname, access='sequential')
        image.write_to_file(image_fname.replace('jpg', 'v'))
