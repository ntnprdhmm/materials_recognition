from PIL import Image, ImageFile
from glob import glob
import sys
import numpy as np

from tqdm import tqdm

from env_functions import get_env
env = get_env()

def ppm_to_jpeg(folder):
    """ Find all .ppm images in the given folder and save them in .jpeg format,
        on the same folder

        Args:
            folder -- string -- path to the folder where to look
    """
    # to fix OSError: image file is truncated
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # for each .ppm image in the folder
    for f in glob(folder + "/*.ppm"):
        # open it as a .ppm file
        im = Image.open(f)
        # save a .jpg copy (change extension from 'ppm' to 'jpg')
        im.save(f[:-3] + 'jpg')

def ppm_to_bin(folder):
    """ Create a bin file from ppm files

        Args:
            folder -- string -- path to the folder where to look
    """
    filenames = glob(folder + "/*.ppm")
    img_size = 1 + 128*128*3
    out = [None] * (img_size * len(filenames)) 
    j = 0
    for i in tqdm(range(len(filenames))):
        im = np.array(Image.open(filenames[i]))

        out[j] = 1 # label
        j += 1

        for k1 in range(3):
            tmp = r = im[:,:,k1].flatten()
            for k2 in range(128*128):
                out[j] = tmp[k2]
                j += 1

    out = np.array(out, np.uint8)
    out.tofile(folder + "/out.bin")

if __name__ == "__main__":
    # if no args provided, nothing to do so leave here
    if len(sys.argv) == 1:
         sys.exit()

    # check witch function is called
    f_called = sys.argv[1]
    if f_called == "ppm_to_jpeg":
        ppm_to_jpeg(env['DATA_DIR'])
    elif f_called == "ppm_to_bin":
        ppm_to_bin(env['DATA_DIR'])
    elif f_called == "ppm_to_bin2":
        ppm_to_bin2(env['DATA_DIR'])