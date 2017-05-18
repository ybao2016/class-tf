import tensorflow as tf

# Create some variables.
v1 = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="v1_name")
v2 = tf.Variable(tf.zeros([200]), name="v2_name")

# Add an op to initialize the variables.
#init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver1 = tf.train.Saver([v1, v2])

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.

save_path = '/tmp/model001.ckpt'

v2_init_op = tf.variables_initializer([v2])
with tf.Session() as sess2:
    #restore v1
    sess2.run(v2_init_op)
    saver1.restore(sess2, save_path)


    print('after restore v1')
    v1_p = sess2.run(v1)[0, 0]
    v2_p = sess2.run(v2)[0]
    print(v1_p)
    print(v2_p)
