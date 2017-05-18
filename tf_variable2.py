import tensorflow as tf

x1 = tf.get_variable('w',initializer=tf.constant(2.0))
x2 = tf.get_variable('x',initializer=tf.constant(5.0))
x3 = tf.square(x1)#tf.constant(4.0)
# 5 x 4 = 20
x2v = tf.assign(x2,x3) # I now expect x2 to be 4.0

x10 = tf.Print(x2,[x2,x3,x2v]) #### first print operation
init_op = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init_op)
print (sess.run(x2)) #### second print operation
print (sess.run(x2))
print(sess.run(x3))
print('x2v.name is', sess.run(x2v.name))
print('x2v.op.name is', sess.run(x2v.op.name))
print (sess.run(x2))
print ('x2v = ', sess.run(x2v))


