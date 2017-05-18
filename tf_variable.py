import tensorflow as tf

# create a variable whose original value is 2
my_var = tf.Variable(10, name="my_var_name")
print (my_var)
# assign a * 2 to a and call that op a_times_two
b = tf.constant(100)
print (b)
c = b + my_var  #is c a op or a tensor. I think both. it's an op first. then the output is a Tensor.
print(c)

my_var_times_two = my_var.assign(2 * my_var)
print(my_var_times_two)

with tf.Session() as sess:
    sess.run(my_var.initializer)
    print(my_var.eval())
    print(sess.run(my_var_times_two)) # >> 4
    print(my_var.eval())
    sess.run(my_var_times_two) # >> 8
    print(my_var.eval())
    sess.run(my_var_times_two) # >> 16
    print(my_var.eval())

    print(c.eval())
    graph_writer = tf.summary.FileWriter('/tmp/model001_tb', sess.graph)
    graph_writer.close()