from __future__ import absolute_import, division, print_function

import pdb
import sys; sys.path.append('./models/slim')
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import tensorflow as tf
import numpy as np

from PIL import Image
from nets.inception_v4 import inception_v4
from nets.inception_utils import inception_arg_scope
from preprocessing import inception_preprocessing
from datasets import imagenet

slim = tf.contrib.slim

height, width = 299, 299
num_classes = 1001

with tf.Graph().as_default():
    im = np.array(Image.open('./example/cat.jpg'))
    image = tf.convert_to_tensor(im)
    processed_image = inception_preprocessing.preprocess_image(image, 
        height, width, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)

    with slim.arg_scope(inception_arg_scope()):
        logits, _ = inception_v4(processed_images, num_classes,
                                       is_training=False)
    predictions = tf.argmax(logits, 1)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(
        'checkpoints/inception_v4.ckpt',
        slim.get_model_variables('InceptionV4'))

    with tf.Session() as sess:
        init_fn(sess)
        pred, prob = sess.run([predictions, probabilities])
        prob = prob[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-prob), key=lambda x:x[1])]

    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (prob[index] * 100, names[index]))