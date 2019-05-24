import argparse
import os
import sys

import tensorflow as tf
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import app
from tensorflow.python.training import saver as saver_lib
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

FLAGS = None

def mlp(_):
  # Parameters
  learning_rate = 0.1
  num_steps = 500
  batch_size = 128
  display_step = 100
  # Network Parameters
  n_hidden_1 = 256 # 1st layer number of neurons
  n_hidden_2 = 256 # 2nd layer number of neurons
  num_input = 784 # MNIST data input (img shape: 28*28)
  num_classes = 10 # MNIST total classes (0-9 digits)
  # tf Graph input
  X = tf.placeholder("float", [None, num_input])
  Y = tf.placeholder("float", [None, num_classes])
  # Store layers weight & bias
  weights = {
      'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
      'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
      'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
  }
  biases = {
      'b1': tf.Variable(tf.random_normal([n_hidden_1])),
      'b2': tf.Variable(tf.random_normal([n_hidden_2])),
      'out': tf.Variable(tf.random_normal([num_classes]))
  }
  # Create model
  def neural_net(x):
      # Hidden fully connected layer with 256 neurons
      layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
      # Hidden fully connected layer with 256 neurons
      layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
      # Output fully connected layer with a neuron for each class
      out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
      return out_layer
  # Construct model
  logits = neural_net(X)
  prediction = tf.nn.softmax(logits)

  # Define loss and optimizer
  loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      logits=logits, labels=Y))
  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
  train_op = optimizer.minimize(loss_op)

  # Evaluate model
  correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

  # Initialize the variables (i.e. assign their default value)
  init = tf.global_variables_initializer()
  # Start training
  with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for step in range(1, num_steps+1):
      batch_x, batch_y = mnist.train.next_batch(batch_size)
      # Run optimization op (backprop)
      sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
      if step % display_step == 0 or step == 1:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                            Y: batch_y})
        print("Step " + str(step) + ", Minibatch Loss= " + \
          "{:.4f}".format(loss) + ", Training Accuracy= " + \
          "{:.3f}".format(acc))
    print("Optimization Finished!")
    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
      sess.run(accuracy, feed_dict={X: mnist.test.images,
                                    Y: mnist.test.labels}))

def write_graph(build_graph, out_dir):
    """Build a graph using build_graph and write it out."""
    g = ops.Graph()
    with g.as_default():
        build_graph(out_dir)
        filename = os.path.join(out_dir, 'test_graph_%s.pb' % build_graph.__name__)
        with open(filename, 'wb') as f:
            f.write(g.as_graph_def().SerializeToString())

def main(_):
    mlp(0)

    # launch the default graph
    sess = tf.Session()

    writer = tf.summary.FileWriter('vis', sess.graph)

    write_graph(mlp, FLAGS.out_dir)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--out_dir',
      type=str,
      default='',
      help='Output directory for graphs, checkpoints and savers.')
  FLAGS, unparsed = parser.parse_known_args()
  app.run(main=main, argv=[sys.argv[0]] + unparsed)
