from PIL import Image, ImageFile
from glob import glob
import sys
import numpy as np

from tqdm import tqdm

from env_functions import get_env
env = get_env()

PVC_LABEL = "pvc"
JOINT_LABEL = "joint"
GLASS_LABEL = "verre"
WOOD_LABEL = "bois"
PE_LABEL = "pe"
PA_LABEL = "pa"
PS_LABEL = "ps"

def labelize(filename, dic):
    """ Given a complete filename, find the label from the basename
        and return a (1 byte) integer to encode this label

        Args:
            filename -- string -- the filename to labelize
            dic -- dictionary -- match a string label to an integer label

        return 1 byte integer 
    """
    str_label = (filename.split("/")[-1].split("_")[0]).lower()
    if str_label in dic:
        return dic[str_label] 
    return 0 # other

def pvc_vs_all():
    return {
        PVC_LABEL: 1
    }

def pvc_joint_glass_wood_other():
    return {
        PVC_LABEL: 1,
        JOINT_LABEL: 2,
        GLASS_LABEL: 3,
        WOOD_LABEL: 4
    }

def pvc_joint_glass_wood_pePaPs_other():
    return {
        PVC_LABEL: 1,
        JOINT_LABEL: 2,
        GLASS_LABEL: 3,
        WOOD_LABEL: 4,
        PE_LABEL: 5,
        PA_LABEL: 5,
        PS_LABEL: 5,
    }

def ppm_to_bin(folder, dic):
    """ Create a bin file from ppm files

        Args:
            folder -- string -- path to the folder where to look
            dic -- func -- dic used to transform filename to 1 byte interger label
    """
    filenames = glob(folder + "/*.ppm")
    #img_size = 1 + 49167# 1 + 128*128*3
    img_size = 1 + 128*128*3
    out = [None] * (img_size * len(filenames))
    j = 0 
    for i in tqdm(range(len(filenames))):
        im = np.array(Image.open(filenames[i]))
        out[j] = labelize(filenames[i], dic)
        j += 1
        for k1 in range(3):
            tmp = im[:,:,k1].flatten()
            for k2 in range(128*128):
                out[j] = tmp[k2]
                j += 1

    nb_images_for_train = int(len(filenames) * (1 - env["TEST_DATA_PERCENTAGE"]))
    end_of_train = nb_images_for_train * img_size

    train_out = np.array(out[:end_of_train], np.uint8)
    train_out.tofile(folder + "/" + env["TRAIN_DATA_FILENAME"])

    test_out = np.array(out[end_of_train:], np.uint8)
    test_out.tofile(folder + "/" + env["TEST_DATA_FILENAME"])

if __name__ == "__main__":
    # if no args provided, nothing to do so leave here
    if len(sys.argv) == 1:
         sys.exit()

    # check witch function is called
    f = sys.argv[1]
    if f == "pvc_vs_all" or f == "1":
        ppm_to_bin(env['DATA_DIR'], pvc_vs_all())
    elif f == "pvc_joint_glass_wood_other" or f == "2":
        ppm_to_bin(env['DATA_DIR'], pvc_joint_glass_wood_other())
    elif f == "pvc_joint_glass_wood_pepaps_other" or f == "3":
        ppm_to_bin(env['DATA_DIR'], pvc_joint_glass_wood_pePaPs_other())