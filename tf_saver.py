import tensorflow as tf

# Create some variables.
v1 = tf.Variable(tf.random_normal([784, 200], stddev=0.35), name="v1_name")
v2 = tf.Variable(tf.zeros([200]), name="v2_name")
v3 = tf.Variable(tf.random_normal([784, 200], stddev=0.35)+3, name="v3_name")

# Add an op to initialize the variables.
update = tf.constant([3], dtype=tf.float32, name="update_const")
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver1 = tf.train.Saver([v1, v2])
print(saver1)

#update_init = tf.variables_initializer([update])

with tf.Session() as sess:
    sess.run(init_op)
    #train_writer = tf.summary.FileWriter('/tmp/model001_tb', sess.graph)

    print('after initialize')
    v1_p0 = sess.run(v1)[0, 0]
    print(v1_p0)
    v2_p0 = sess.run(v2)[0]
    print(v2_p0)


    # update v1

    v1_op = v1.assign(v3+10)
    #print v2
    print('v2 is ', v2)


    #v1 = v1 + 3.0
    #v2 += 2
    v2_op = v2.assign(v2+2)
    print('v2 is ', v2)

    print('after update')
    sess.run(v1_op)
    sess.run(v2_op)
    v1_p = sess.run(v1)[0, 0]
    v2_p = sess.run(v2)[0]
    print(v1_p)
    print(v2_p)


    # Save the variables to disk.

    save_path = saver1.save(sess, "/tmp/model001.ckpt")
    print ("Model saved in file: ", save_path)

    train_writer = tf.summary.FileWriter('/tmp/model001_tb', sess.graph)

    train_writer.close()

