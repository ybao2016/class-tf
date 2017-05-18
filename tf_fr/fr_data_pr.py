"""
use TFRecord to preprocess data
Nick Bao

endode --utf-8

"""
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

# image supposed to have shape: 480 x 640 x 3 = 921600
IMAGE_PATH = '/home/nick/datasets/myphoto'

def get_image_filepath_label(file_dir_root):
    counter = 0
    directories = []
    class_names = []
    for filename in os.listdir(file_dir_root):
        path = os.path.join(file_dir_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)
    dir_to_class = dict(zip(directories, class_names))
    photo_filenames = []
    photo_labels = []

    for directory in directories:
        label = dir_to_class[directory]
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)
            photo_labels.append(label)

            counter += 1
    print('add %d photo files from directory %s' %(counter, file_dir_root))

    return photo_filenames, photo_labels, sorted(class_names)

def get_image_binary(filename):
    """ You can read in the image using tensorflow too, but it's a drag
        since you have to create graphs. It's much easier using Pillow and NumPy
    """
    image = Image.open(filename)
    image = np.asarray(image, np.uint8)
    shape = np.array(image.shape, np.int32)
    return shape.tobytes(), image.tobytes() # convert image to raw data bytes in the array.

def write_to_tfrecord(label, shape, binary_image, tfrecord_file, idx, writer):
    """ This example is to write a sample to TFRecord file. If you want to write
    more samples, just use a loop.
    """
    # write label, shape, and image content to the TFRecord file
    example = tf.train.Example(features=tf.train.Features(feature={
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label])),
                'shape': tf.train.Feature(bytes_list=tf.train.BytesList(value=[shape])),
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[binary_image])),
                'idx': tf.train.Feature(int64_list=tf.train.Int64List(value=[idx]))
                }))
    writer.write(example.SerializeToString())


def create_tfrecord(photo_filenames, photo_labels, class_names, tfrecord_file):
    class_name_to_ids = dict(zip(class_names, range(len(class_names))))
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
        for i in range(len(photo_filenames)):
            image_file = photo_filenames[i]
            try:
                shape, binary_image = get_image_binary(image_file)
            except:
                print('can not read image file %s by Pillow, continue', image_file)
                continue
            label = photo_labels[i]
            idx = class_name_to_ids[label]
            label = label.encode('utf-8')
            write_to_tfrecord(label, shape, binary_image, tfrecord_file, idx, writer)

def read_from_tfrecord(filenames):
    tfrecord_file_queue = tf.train.string_input_producer(filenames, name='queue')
    reader = tf.TFRecordReader()
    _, tfrecord_serialized = reader.read(tfrecord_file_queue)

    # label and image are stored as bytes but could be stored as
    # int64 or float64 values in a serialized tf.Example protobuf.
    tfrecord_features = tf.parse_single_example(tfrecord_serialized,
                        features={
                            'label': tf.FixedLenFeature([], tf.string),
                            'shape': tf.FixedLenFeature([], tf.string),
                            'image': tf.FixedLenFeature([], tf.string),
                            'idx': tf.FixedLenFeature([], tf.int64),
                        }, name='features')
    # image was saved as uint8, so we have to decode as uint8.
    image = tf.decode_raw(tfrecord_features['image'], tf.uint8)
    shape = tf.decode_raw(tfrecord_features['shape'], tf.int32)
    # label = tf.decode_raw(tfrecord_features['label'], tf.int32)
    # the image tensor is flattened out, so we have to reconstruct the shape
    image = tf.reshape(image, shape)
    label = tf.cast(tfrecord_features['label'], tf.string)
    idx = tf.cast(tfrecord_features['idx'], tf.int64)

    return label, shape, image, idx

def read_tfrecord(tfrecord_file):
    label, shape, image, idx = read_from_tfrecord([tfrecord_file])

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            for i in range(19) :
                label_out, image_out, shape_out, idx_out = sess.run([label, image, shape, idx])
                print('idx is ', idx_out)
                print('label is ', label_out.decode('utf-8'))
        except tf.errors.OutOfRangeError as e:
            coord.request_stop(e)

        finally:
            coord.request_stop()
            coord.join(threads)

        # plt.imshow(image)
        # plt.title(label.decode('utf-8'))
        # plt.show()

def main():
    # assume the image has the label Chihuahua.
    # in practice, you'd want to use binary numbers for your labels to save space

    train_name = 'train'
    test_name = 'test'
    # cata_names[train, nam]
    train_root_path = os.path.join(IMAGE_PATH, train_name)

    photo_filenames, photo_labels, class_names = get_image_filepath_label(train_root_path)

    tfrecord_file = os.path.join(IMAGE_PATH, train_root_path,  train_name + '001.tfrecord')

    create_tfrecord(photo_filenames, photo_labels, class_names, tfrecord_file)
    print('photo labels are: ', photo_labels)


    read_tfrecord(tfrecord_file)

if __name__ == '__main__':
    main()

