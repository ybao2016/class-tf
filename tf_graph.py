import tensorflow as tf
a = tf.constant(2, dtype=tf.float32, name="a")
b = tf.constant(3, dtype=tf.float32, name="b")

x = tf.add(a, b, name="add")
y = tf.multiply(a, b, name="mul")

with tf.Session() as sess:
    print(sess.run(x))
    print(sess.run(y))
    print
    print(sess.graph.as_graph_def())
    train_writer = tf.summary.FileWriter('/tmp/model001_tb', sess.graph)
    train_writer.close()

