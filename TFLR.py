#！/user/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Mingqi, Yuan'

import tensorflow as tf
import matplotlib.pyplot as plt

def LR(input_x, input_y, dimensions, log_dir, learning_rate, train_steps):
    with tf.name_scope('Input'):
        x_data = tf.placeholder(tf.float32, [None, dimensions], name='x_data')
        y_data = tf.placeholder(tf.float32, [None, 1], name='y_data')
    with tf.name_scope('Weight And Bias'):
        weight = tf.Variable(tf.truncated_normal([dimensions, 1]), name='weight')
        bias = tf.Variable(tf.truncated_normal([dimensions]), name='bias')
        tf.summary.histogram('The change of the weight', weight)
        tf.summary.histogram('The change of the bias', bias)
    with tf.name_scope('Output'):
        y_model = tf.add(tf.matmul(x_data, weight), bias)
    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.pow((y_data - y_model), 2))
        tf.summary.scalar('The change of the loss', loss)
    with tf.name_scope('Train'):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        merged = tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        data = []
        print("Start training!")
        for i in range(train_steps):
            sess.run(train_op, feed_dict={x_data : input_x, y_data: input_y})
            c = sess.run(loss, feed_dict={x_data : input_x, y_data: input_y})
            data.append(c)
            print("Step: %d, loss = %.4f, weight = %.4f, bias = %.4f" % (i+1, c, sess.run(weight), sess.run(bias)))
            if i % 10 == 0:
                summary, lo = sess.run([merged, loss], feed_dict={x_data : input_x, y_data: input_y})
                writer.add_summary(summary, i)
        final_loss = sess.run(loss, feed_dict={x_data : input_x, y_data: input_y})
        weight, bias = sess.run([weight, bias])
        print("The final loss:", final_loss)
        print("The weight:", weight)
        print("The bias:", bias)
        writer.close()
        plt.plot(data, 'red')
        plt.title('The change of the loss')
        plt.xlabel('The sampling point')
        plt.ylabel('Loss')
        plt.tight_layout()
        plt.show()

def main():
    print("Test passed!")
if __name__ == '__main__':
    main()



