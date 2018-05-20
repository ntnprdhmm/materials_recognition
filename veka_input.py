import tensorflow as tf
from glob import glob

from env_functions import get_env
env = get_env()

def read_veka(filename_queue, label_fifo):
    """Reads and parses examples.

    Recommendation: if you want N-way read parallelism, call this function
    N times.  This will give you N independent Readers reading different
    files & positions within those files, which will give better mixing of
    examples.

    Args:
      filename_queue: A queue of strings with the filenames to read from.
      label_fifo -- FIFOQueue -- A queue with all the labels

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

    # get the matching label from label_fifo
    result.label = label_fifo.dequeue()
    tf.reshape(result.label, [1])

    # Decode the image as a JPEG file, this will turn it into a Tensor of int8
    # which we can then use in training.
    result.uint8image = tf.image.decode_jpeg(image_file)

    return result

def label_from_filename(filename):
    """ Parse the filename to find the label

        Args:
            filename -- string -- the filename to parse

        return a integer
    """
    label_string = filename.split('/')[-1].split('_')[0]
    label = 1 if label_string == 'PVC' else 0
    return label

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
    """Construct a queued batch of images and labels.

        Args:
            image: 3-D Tensor of [height, width, 3] of type.float32.
            label: 1-D Tensor of type.int32
            min_queue_examples: int32, minimum number of samples to retain
                in the queue that provides of batches of examples.
            batch_size: Number of images per batch.
            shuffle: boolean indicating whether to use a shuffling queue.

        Returns:
            images: Images. 4D tensor of [batch_size, height, width, 3] size.
            labels: Labels. 1D tensor of [batch_size] size.
    """
    # Create a queue that shuffles the examples, and then
    # read 'batch_size' images + labels from the example queue.
    num_preprocess_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    tf.summary.image('images', images)

    return images, tf.reshape(label_batch, [batch_size])

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

    # extract the labels of each filename in a new list
    labels = labels = [label_from_filename(f) for f in filenames]
    lv = tf.constant(labels)
    label_fifo = tf.FIFOQueue(len(filenames), tf.int32, shapes=[[]])
    label_enqueue = label_fifo.enqueue_many([lv])

    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    with tf.name_scope('data_augmentation'):
        # Read examples from files in the filename queue.
        read_input = read_veka(filename_queue, label_fifo)

        reshaped_image = tf.cast(read_input.uint8image, tf.float32)

        height = 128
        width = 128

        # Image processing for training the network. Note the many random
        # distortions applied to the image.

        # Randomly crop a [height, width] section of the image.
        distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

        # Randomly flip the image horizontally.
        distorted_image = tf.image.random_flip_left_right(distorted_image)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        # NOTE: since per_image_standardization zeros the mean and makes
        # the stddev unit, this likely has no effect see tensorflow#1458.
        distorted_image = tf.image.random_brightness(distorted_image,
                                                     max_delta=63)
        distorted_image = tf.image.random_contrast(distorted_image,
                                                   lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels.
        float_image = tf.image.per_image_standardization(distorted_image)

        # Set the shapes of tensors.
        float_image.set_shape([height, width, 3])
        #read_input.label.set_shape([1])

        # Ensure that the random shuffling has good mixing properties.
        min_fraction_of_examples_in_queue = 0.4
        NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50
        min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                                 min_fraction_of_examples_in_queue)
        print ('Filling queue with %d CIFAR images before starting to train. '
               'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=True)

if __name__ == '__main__':
    distorted_inputs(env['TRAIN_DIR'], env['BATCH_SIZE'])
