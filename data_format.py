from PIL import Image, ImageFile
from glob import glob
import sys

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

if __name__ == "__main__":
    # if no args provided, nothing to do so leave here
    if len(sys.argv) == 1:
         sys.exit()

    # check witch function is called
    f_called = sys.argv[1]
    if f_called == "ppm_to_jpeg":
        ppm_to_jpeg(env['DATA_DIR'])
