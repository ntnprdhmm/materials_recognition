import tensorflow as tf
from glob import glob

from env_functions import get_env
env = get_env()

def read_veka(filename_queue):
    """Reads and parses examples.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
      filename_queue: A queue of strings with the filenames to read from.

    Returns:
      An object representing a single example, with the following fields:
        height: number of rows in the result (128)
        width: number of columns in the result (128)
        depth: number of color channels in the result (3)
        key: a scalar string Tensor describing the filename & record number
          for this example.
        label: an int32 Tensor with the label in the range 0..1.
        uint8image: a [height, width, depth] uint8 Tensor with the image data
    """

    class VEKARecord(object):
      pass
    result = VEKARecord()

    result.height = 128
    result.width = 128
    result.depth = 3

    # will read images
    image_reader = tf.WholeFileReader()

    # Read a whole file from the queue
    result.key, image_file = image_reader.read(filename_queue)

    # Parse result.key (the filename) to find the label
    # TODO : fix label issue
    #label_string = result.key.split('/')[-1].split('_')[0]
    #result.label = tf.cast([0, 1] if label_string == 'PVC' else [1, 0], tf.int32)

    # Decode the image as a JPEG file, this will turn it into a Tensor of int8
    # which we can then use in training.
    result.uint8image = tf.image.decode_jpeg(image_file)

    return result

def distorted_inputs(data_dir, batch_size):
    """Construct distorted input for CIFAR training using the Reader ops.

        Args:
            data_dir: Path to the train data directory (where are the .jpg).
            batch_size: Number of images per batch.

        Returns:
            images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
            labels: Labels. 1D tensor of [batch_size] size.
    """
    # create a list with all the jpg files of the given folder (data_dir)
    filenames = glob(data_dir + '/*.jpg')
    for f in filenames:
       if not tf.gfile.Exists(f):
           raise ValueError('Failed to find file: ' + f)

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    with tf.name_scope('data_augmentation'):
        # Read examples from files in the filename queue.
        read_input = read_veka(filename_queue)

if __name__ == '__main__':
    distorted_inputs(env['TRAIN_DIR'], env['BATCH_SIZE'])
