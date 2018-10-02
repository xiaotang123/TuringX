#！/user/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Mingqi, Yuan'
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

def LGR(input_x, input_y, factors, classes, learning_rate, train_steps, log_dir):
    with tf.name_scope('Input'):
        x_data = tf.placeholder(tf.float32, [None, factors])
        y_data = tf.placeholder(tf.float32, [None, classes])

    with tf.name_scope('Weight And Bias'):
        weight = tf.Variable(tf.truncated_normal([factors, classes]), dtype= tf.float32)
        bias = tf.Variable(tf.truncated_normal([1, classes]), dtype= tf.float32)

    with tf.name_scope('Output'):
        output = tf.nn.softmax(tf.matmul(x_data, weight) + bias)

    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = output, logits = y_data))

    with tf.name_scope('Train'):
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        # initialize the variable
        sess.run(tf.global_variables_initializer()) # 开始训练
        print("Start training!")
        lo = []
        sample = np.arange(train_steps) # 训练train_steps次
        for i in range(train_steps):
            for (x,y) in zip(input_x, input_y):
                z1 = x.reshape(1, factors)
                z2 = y.reshape(1, classes)
                sess.run(train_op, feed_dict = {x_data : z1, y_data : z2})
            l = sess.run(loss, feed_dict = {x_data : z1, y_data : z2})
            lo.append(l)
        print('The weight matrix and the bias matrix:')
        print(weight.eval(sess))
        print(bias.eval(sess))
        writer.close()
        # plot the variation of the loss
        plt.plot(sample, lo, color="red", linewidth = '1')
        plt.title("The variation of the loss")
        plt.xlabel("Sampling Point")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

def main():
    print('Test passes!')
if __name__ == '__main__':
    main()