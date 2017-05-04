from __future__ import absolute_import, division

import pdb
import sys; sys.path.append('./models/slim')
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
from nets.inception_v4 import inception_v4
from nets.inception_utils import inception_arg_scope

slim = tf.contrib.slim

height, width = 299, 299
num_classes = 1001
c1, c5 = 0, 0
lines = np.loadtxt('imagenet/val.txt', str, delimiter='\n')

with tf.Graph().as_default():
    eval_inputs = tf.placeholder(tf.float32, [1, height, width, 3])
    with slim.arg_scope(inception_arg_scope()):
        logits, _ = inception_v4(eval_inputs, num_classes,
                                       is_training=False)
    predictions = tf.argmax(logits, 1)
    probabilities = tf.nn.softmax(logits)

    init_fn = slim.assign_from_checkpoint_fn(
        'checkpoints/inception_v4.ckpt',
        slim.get_model_variables('InceptionV4'))

    sess = tf.Session()
    init_fn(sess)

    for idx, line in enumerate(lines):
        
        [imname, label] = line.split(' ')
        label = int(label) + 1
        im = np.array(Image.open('imagenet/ILSVRC2012_img_val/' + imname).convert('RGB'))
        processed_image = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
        processed_image = (processed_image.astype(np.float32) / 256 - 0.5) * 2
        processed_images = np.expand_dims(processed_image, axis=0)
        pred, prob = sess.run([predictions, probabilities], feed_dict={
                eval_inputs: processed_images
            })
        prob = prob[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-prob), key=lambda x:x[1])]

        top1 = sorted_inds[0]
        top5 = sorted_inds[0:5]
        if label == top1:
            c1 += 1
        if label in top5:
            c5 += 1
        print('images: %d\ttop 1: %0.4f\ttop 5: %0.4f' % (idx + 1, c1/(idx + 1), c5/(idx + 1)))
        
