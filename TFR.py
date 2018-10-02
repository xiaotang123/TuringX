#！/user/bin/env python3
# -*- coding:utf-8 -*-

__author__ = 'Mingqi, Yuan'

import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt

class generate:
    def graph(self):
        # input the processing demand
        file_dir = input("The dir of the pictures:")
        c = input("The classes:")
        classes = c.split(",")
        save_dir = input("The filename and dir to save:")
        shape = int(input("The shape after treatment:"))
        add = file_dir

        writer = tf.python_io.TFRecordWriter(save_dir)
        for index, name in enumerate(classes):
            # write the pictures' directory
            class_path = add + name + '/'
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name
                Img = Image.open(img_path)
                img = Img.resize((shape, shape))
                # transform the graph into binary format
                img_raw = img.tobytes()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                        }))
                writer.write(example.SerializeToString())
        writer.close()
        print("Generation accomplished!")
        print("Your TFRecords exists in", save_dir)

class read:
    def graph(self, filename, width, height, channels, flag):
        filename_queue = tf.train.string_input_producer([filename])

        reader = tf.TFRecordReader()
        # return the file and the filename
        _, serialized_example = reader.read(filename_queue)
        # fetch the image data and the label
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'img_raw': tf.FixedLenFeature([], tf.string)
                                           })
        img = tf.decode_raw(features['img_raw'], tf.uint8)
        img = tf.reshape(img, [width, height, channels])
        # output the tensor 'img' and the tensor 'label'
        img = tf.cast(img, tf.float32)
        label = tf.cast(features['label'], tf.int32)

        # flag = 1:show the result, flag = 0: not
        if flag == 1:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                print("The shape and the label:")
                for i in range(10):
                    example, l = sess.run([img, label])
                    print(example.shape, l)
                    plt.imshow(example)
                    plt.show()

                coord.request_stop()
                coord.join(threads)
        if flag == 0:
            pass

        return img, label

def main():
    print("Test passed!")
if __name__ == '__main__':
    main()



