import tensorflow as tf
import numpy as np

learning_rate = 0.003

n_input = 27 # alphabet and space
n_hidden = 128 # hidden layer features
max_sequence_length = 11
alphabet = 'abcdefghijklmnopqrstuvwxyz '
ethnicities = ['chinese', 'japanese', 'vietnamese', 'korean']
n_classes = len(ethnicities)

def __main__():
  str_list = []
  names_list = []
  ethnicity_list = []
  with open('names.csv', 'r') as csv:
    for line in csv:
      l = [s.strip() for s in line.split(',')]
      if(l[1] in ethnicities):
        names_list.append(name_one_hot(l[0], max_sequence_length))
        ethnicity_list.append(ethnicity_one_hot(l[1]))
  rng_state = np.random.get_state() # use the same random number generator state
  np.random.shuffle(names_list)     # when shuffling the two lists
  np.random.set_state(rng_state)    # they are effectively shuffled in parallel so that inputs still correspond to outputs after shuffling
  np.random.shuffle(ethnicity_list)

  size = len(names_list)
  training_X = np.array(names_list[:size*2/3])
  training_y = np.array(ethnicity_list[:size*2/3])
  testing_X = np.array(names_list[size*2/3:])
  testing_y = np.array(ethnicity_list[size*2/3:])

  X = tf.placeholder(tf.float32, [None, max_sequence_length, n_input])
  y = tf.placeholder(tf.float32, [None, n_classes])

  out_weights = weight_variable([n_hidden, n_classes])
  out_biases = bias_variable([n_classes])

  rnn_cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
  outputs, states = tf.nn.dynamic_rnn(rnn_cell, X, dtype=tf.float32)

  y_ = tf.matmul(outputs[:,-1,:], out_weights) + out_biases # predict y based on final rnn output

  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
  train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
  
  correct_prediction = tf.equal(tf.argmax(y_,1), tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  init = tf.global_variables_initializer()

  sess = tf.InteractiveSession()
  sess.run(init)
  for _ in range(200):
    sess.run(train_step, feed_dict={X: training_X, y: training_y})
    if _%10 == 0:
      train_accuracy = accuracy.eval(feed_dict={
        X:training_X, y:training_y})
      print("step %d, training accuracy %g"%(_, train_accuracy))
      test_accuracy = accuracy.eval(feed_dict={X:testing_X, y:testing_y})
      print("testing accuracy", test_accuracy)


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def name_one_hot(name, max_sequence_length):
  result = []
  for char in name:
    v = np.zeros(27, dtype=np.int) # count space as a character
    v[alphabet.index(char)] = 1
    result.append(v)
  while len(result) < max_sequence_length:
    result.append(np.zeros(27, dtype=np.int))
  result = np.array(result)
  return result

def ethnicity_one_hot(ethnicity):
  v = np.zeros(n_classes, dtype=np.int)
  v[ethnicities.index(ethnicity)] = 1
  return v


__main__()