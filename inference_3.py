import tensorflow as tf

import veka

from env_functions import get_env
env = get_env()

NUM_CLASSES = len(env['DATASET_TYPE'].split('_'))
SHAPE_FINAL = int((env["IMAGE_SIZE"]/(env["POOL_STEP"]**env["NB_CONV_LAYERS"]))*env["CONV_OUTPUT"])
LOCAL_OUTOUT = int(SHAPE_FINAL/env["LOCAL_OUTPUT_FACTOR"])

def inference(images):
  """Build the PVC vs wood vs glass vs joint vs PE/PA/PS vs other model.

  Args:
    images: Images returned from distorted_inputs() or inputs().

  Returns:
    Logits.
  """
  print("INFERENCE: PVC vs woord vs glass vs joint vs PE/PA/PS vs other")
  # We instantiate all variables using tf.get_variable() instead of
  # tf.Variable() in order to share variables across multiple GPU training runs.
  # If we only ran this model on a single GPU, we could simplify this function
  # by replacing all instances of tf.get_variable() with tf.Variable().
  #
  # conv1
  with tf.variable_scope('conv1') as scope:
    kernel = veka._variable_with_weight_decay('weights',
                                         shape=[env["CONV_GRID_LEN"], env["CONV_GRID_LEN"], 3, env["CONV_OUTPUT"]],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(images, kernel, [env["CONV_STEP"], env["CONV_STEP"], env["CONV_STEP"], 1], padding='SAME')
    biases = veka._variable_on_cpu('biases', [env["CONV_OUTPUT"]], tf.constant_initializer(0.0))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(pre_activation, name=scope.name)
    veka._activation_summary(conv1)

  # pool1
  pool1 = tf.nn.max_pool(conv1, 
                         ksize=[1, env["POOL_GRID_LEN"], env["POOL_GRID_LEN"], 1], 
                         strides=[1, env["POOL_STEP"], env["POOL_STEP"], 1],
                         padding='SAME', name='pool1')
  # norm1
  norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm1')

  # conv2
  with tf.variable_scope('conv2') as scope:
    kernel = veka._variable_with_weight_decay('weights',
                                         shape=[env["CONV_GRID_LEN"], env["CONV_GRID_LEN"], env["CONV_OUTPUT"], env["CONV_OUTPUT"]],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(norm1, kernel, [1, env["CONV_STEP"], env["CONV_STEP"], 1], padding='SAME')
    biases = veka._variable_on_cpu('biases', [env["CONV_OUTPUT"]], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(pre_activation, name=scope.name)
    veka._activation_summary(conv2)

  # norm2
  norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm2')
  # pool2
  pool2 = tf.nn.max_pool(norm2, ksize=[1, env["POOL_GRID_LEN"], env["POOL_GRID_LEN"], 1], 
                         strides=[1, env["POOL_STEP"], env["POOL_STEP"], 1],
                         padding='SAME', name='pool2')
  
  # conv3
  with tf.variable_scope('conv3') as scope:
    kernel = veka._variable_with_weight_decay('weights',
                                         shape=[env["CONV_GRID_LEN"], env["CONV_GRID_LEN"], env["CONV_OUTPUT"], env["CONV_OUTPUT"]],
                                         stddev=5e-2,
                                         wd=None)
    conv = tf.nn.conv2d(pool2, kernel, [1, env["CONV_STEP"], env["CONV_STEP"], 1], padding='SAME')
    biases = veka._variable_on_cpu('biases', [env["CONV_OUTPUT"]], tf.constant_initializer(0.1))
    pre_activation = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(pre_activation, name=scope.name)
    veka._activation_summary(conv3)

  # norm3
  norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                    name='norm3')
  # pool3
  pool3 = tf.nn.max_pool(norm3, ksize=[1, env["POOL_GRID_LEN"], env["POOL_GRID_LEN"], 1], 
                         strides=[1, env["POOL_STEP"], env["POOL_STEP"], 1],
                         padding='SAME', name='pool3') 

  # local4
  with tf.variable_scope('local4') as scope:
    # Move everything into depth so we can perform a single matrix multiply.
    reshape = tf.reshape(pool3, [images.get_shape().as_list()[0], -1])
    dim = reshape.get_shape()[1].value
    weights = veka._variable_with_weight_decay('weights', 
                                          shape=[dim, SHAPE_FINAL],
                                          stddev=0.04, wd=0.004)
    biases = veka._variable_on_cpu('biases', [SHAPE_FINAL], tf.constant_initializer(0.1))
    local4 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    veka._activation_summary(local4)

  # local5
  with tf.variable_scope('local5') as scope:
    weights = veka._variable_with_weight_decay('weights', 
                                          shape=[SHAPE_FINAL, LOCAL_OUTOUT],
                                          stddev=0.04, wd=0.004)
    biases = veka._variable_on_cpu('biases', [LOCAL_OUTOUT], tf.constant_initializer(0.1))
    local5 = tf.nn.relu(tf.matmul(local4, weights) + biases, name=scope.name)
    veka._activation_summary(local5)

  # linear layer(WX + b),
  # We don't apply softmax here because
  # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
  # and performs the softmax internally for efficiency.
  with tf.variable_scope('softmax_linear') as scope:
    weights = veka._variable_with_weight_decay('weights', [LOCAL_OUTOUT, NUM_CLASSES],
                                          stddev=1/LOCAL_OUTOUT, wd=None)
    biases = veka._variable_on_cpu('biases', [NUM_CLASSES],
                              tf.constant_initializer(0.0))
    softmax_linear = tf.add(tf.matmul(local5, weights), biases, name=scope.name)
    veka._activation_summary(softmax_linear)

  return softmax_linear