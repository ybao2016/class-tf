import time
start_time = time.time()

import numpy as np
import tensorflow as tf


# Model parameters
W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)
# loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
# optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
# training data
x_train = [1,2,3,4]
y_train = [0,-1,-2,-3]
# training loop
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init) # reset values to wrong

#print tensors value out
print(sess.run(W))
#tf.Print(W, [W], message="this message")
#W.eval(sess)

loop_count = 1000
for i in range(loop_count):
  sess.run(train, {x:x_train, y:y_train})

# evaluate training accuracy
curr_W, curr_b, curr_loss  = sess.run([W, b, loss], {x:x_train, y:y_train})
print("W: %f b: %s loss: %s"%(curr_W, curr_b, curr_loss))

from pylab import *
figure(figsize=(8,6), dpi=80)
subplot(1, 1, 1)
plot(x_train, y_train, 'r+')
x_re = range(-1, 7, 1)
y_re = curr_W * x_re + curr_b
plot(x_re, y_re, 'b')
show()

# your code
elapsed_time = time.time() - start_time
print('train %d loop with linear model used %3.6f' % (loop_count, elapsed_time))