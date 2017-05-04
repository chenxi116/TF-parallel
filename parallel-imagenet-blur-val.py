from __future__ import absolute_import, division

import pdb
import sys; sys.path.append('./models/slim')
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image
from PIL import ImageFilter
from nets.inception_v4 import inception_v4
from nets.inception_utils import inception_arg_scope

slim = tf.contrib.slim

height, width = 299, 299
num_classes = 1001
c1, c5 = 0, 0
lines = np.loadtxt('imagenet/val.txt', str, delimiter='\n')

# Gaussian filter instance
flt = ImageFilter.ModeFilter(size=3)

# tic toc helper
import time
class tictoc:
    def __init__(self, name=None):
        self.name = name
    def __enter__(self):
        self.tstart = time.time()
    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)

# multi-threading environment
from multiprocessing import Process, Queue, Lock
from math import floor
def fetch_img(prc_i, num_t, img_q):
    lines = np.loadtxt('imagenet/val.txt', str, delimiter='\n')
    steps = int(floor(lines.size / num_t))
    start = steps * prc_i
    end = start + steps
    flt = ImageFilter.ModeFilter(size=3)
    if prc_i == num_t - 1:
        end = lines.size
    for i in range(start, end):
        imname, label = lines[i].split(' ')
        label = int(label) + 1
        im = Image.open('imagenet/ILSVRC2012_img_val/' + imname).convert('RGB')
        im = np.array(im.filter(flt))
        processed_image = cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA)
        processed_image = (processed_image.astype(np.float32) / 256 - 0.5) * 2
        processed_images = np.expand_dims(processed_image, axis=0)
        img_q.put([processed_images, label])

img_q = Queue(maxsize=256)
num_t = int(sys.argv[2])
prc_l = []
for i in range(num_t):
    prc_l.append(Process(target=fetch_img, args=(i, num_t, img_q)))
    prc_l[-1].start()

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
    img_cnt = 0
    img_tlt = lines.size

    while img_cnt < img_tlt:
        processed_images, label = img_q.get()
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
        idx = img_cnt
        print('images: %d\ttop 1: %0.4f\ttop 5: %0.4f' % (idx + 1, c1/(idx + 1), c5/(idx + 1)))
        img_cnt += 1

for i in range(num_t):
    prc_l[i].join()
