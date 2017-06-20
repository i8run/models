# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Timing benchmark for Inception_V1 inference.

To run, use:
  bazel run -c opt --config=cuda \
      models/tutorials/image/alexnet:alexnet_benchmark

Across 100 steps on batch size = 128.

Forward pass:
Run on Tesla K40c: 145 +/- 1.5 ms / batch
Run on Titan X:     70 +/- 0.1 ms / batch

Forward-backward pass:
Run on Tesla K40c: 480 +/- 48 ms / batch
Run on Titan X:    244 +/- 30 ms / batch
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import math
import sys
import time

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
sys.path.append("/home/arda/yuan.liu/models/slim")
from nets import inception_v1

FLAGS = None

def loss(logits, labels):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, labels], 1)
    onehot_labels = tf.sparse_to_dense(
        concated, tf.stack([batch_size, 1000]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=onehot_labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())

def time_tensorflow_run(session, target, info_string):
  """Run the computation to obtain the target tensor and print timing stats.

  Args:
    session: the TensorFlow session to run the computation under.
    target: the target Tensor that is passed to the session's run() function.
    info_string: a string summarizing this run, to be printed with the stats.

  Returns:
    None
  """
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  for i in xrange(FLAGS.num_batches + num_steps_burn_in):
    start_time = time.time()
    _ = session.run(target)
    duration = time.time() - start_time
    if i >= num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print ('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
         (datetime.now(), info_string, FLAGS.num_batches, mn, sd))

def run_benchmark():
  """Run the benchmark on Inception_V1."""
  with tf.Graph().as_default():
    # Generate some dummy images.
    image_size = 224
    # Note that our padding definition is slightly different the cuda-convnet.
    # In order to force the model to start with the same activations sizes,
    # we add 3 to the image_size and employ VALID padding above.
    images = tf.Variable(tf.random_normal([FLAGS.batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1), trainable=False)

    labels = tf.Variable(tf.ones([FLAGS.batch_size],
                                 dtype=tf.int32), trainable=False)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits, end_points = inception_v1.inception_v1(images)
    # Build an initialization operation.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph.
    config = tf.ConfigProto()
    # config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    sess.run(init)

    # Run the forward benchmark.
    time_tensorflow_run(sess, logits, "Forward")

    # Add a simple objective so we can calculate the backward pass.
    objective = loss(logits, labels)

    # Variables to train.
    variables_to_train = tf.trainable_variables()

    # Compute the gradient with respect to all the parameters.
    grad = tf.gradients(objective, variables_to_train)
    # Run the backward benchmark.
    time_tensorflow_run(sess, grad, "Forward-backward")


def main(_):
  run_benchmark()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='Batch size.'
  )
  parser.add_argument(
      '--num_batches',
      type=int,
      default=100,
      help='Number of batches to run.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
