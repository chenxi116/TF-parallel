from __future__ import absolute_import, division

import pdb
import sys; sys.path.append('./models/slim')
import os; os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
import tensorflow as tf
import numpy as np
try:
    import cv2
    imresize = lambda x, dim: cv2.resize(x, dim, interpolation=cv2.INTER_AREA) 
except:
    import scipy
    imresize = lambda x, dim: scipy.misc.imresize(x, dim)

from PIL import Image
from PIL import ImageFilter
from nets.inception_v4 import inception_v4
from nets.inception_utils import inception_arg_scope
from collections import namedtuple

OP = namedtuple('OP', 'input pred prob')

slim = tf.contrib.slim

height, width = 299, 299
num_classes = 1001
c1, c5 = 0, 0
lines = np.loadtxt('imagenet/val.txt', str, delimiter='\n')
nb_gpus = len(sys.argv[1].split(','))

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
from multiprocessing.queues import SimpleQueue
from math import floor
fetch_test = False
def fetch_img(prc_i, num_t, img_q):
    print('Thread %d start!!!' % prc_i)
    lines = np.loadtxt('imagenet/val.txt', str, delimiter='\n')
    steps = int(floor(lines.size / num_t))
    start = steps * prc_i
    end = start + steps
    flt = ImageFilter.ModeFilter(size=3)
    if prc_i == num_t - 1:
        end = lines.size
    for i in range(start, end):
        print('Thread %d start loop' % prc_i)
        imname, label = lines[i].split(' ')
        label = int(label) + 1
        print('Thread %d load image' % prc_i)
        im = Image.open('imagenet/ILSVRC2012_img_val/' + imname).convert('RGB')
        print('Thread %d filter image' % prc_i)
        im = np.array(im.filter(flt))
        print('Thread %d proprocess image' % prc_i)
        processed_image = imresize(im, (width, height))
        processed_image = (processed_image.astype(np.float32) / 256 - 0.5) * 2
        processed_images = np.expand_dims(processed_image, axis=0)
        print('Thread %d try to enqueue' % prc_i)
        img_q.put([processed_images, label])
        print('Thread %d successfully enqueue' % prc_i)

# img_q = Queue(maxsize=1024)
img_q = SimpleQueue()
num_t = int(sys.argv[2])
prc_l = []
for i in range(num_t):
    print('fetch thread %d' % i)
    prc_l.append(Process(target=fetch_img, args=(i, num_t, img_q)))
    prc_l[-1].daemon=True
    prc_l[-1].start()

def eval(rank, img_q, sess, op, img_tlt):
    img_cnt = 0
    while img_cnt < img_tlt:
        print('wait for image')
        processed_images, label = img_q.get()
        print('got image')
        if fetch_test:
            img_cnt += 1
            sys.stdout.write('\r{:07d}/{:07d}'.format(img_cnt, img_tlt))
            sys.stdout.flush()
            continue
        pred, prob = sess.run([op.pred, op.prob], feed_dict={
                op.input: processed_images
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
        print('gpu %d images: %d\ttop 1: %0.4f\ttop 5: %0.4f' % (rank, idx + 1, c1/(idx + 1), c5/(idx + 1)))
        img_cnt += 1

with tf.Graph().as_default():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.log_device_placement = True
    sess = tf.Session(config=config)
    subgraph = []
    for gpu in range(nb_gpus):
        print('loading graph on /gpu:%d' % gpu)
        eval_input = tf.placeholder(tf.float32, [1, height, width, 3])
        with tf.device('/gpu:%d' % gpu):
            with tf.name_scope('tower-%d' % gpu):
                with slim.arg_scope(inception_arg_scope()):
                    resue = None if gpu == 0 else True
                    logits, _ = inception_v4(eval_input, num_classes, reuse=resue,
                                             scope='InceptionV4',
                                             is_training=False)
                    predictions = tf.argmax(logits, 1)
                probabilities = tf.nn.softmax(logits)
                subgraph.append(OP(eval_input, predictions, probabilities))

    init_fn = slim.assign_from_checkpoint_fn(
        'checkpoints/inception_v4.ckpt',
        slim.get_model_variables('InceptionV4'))
    init_fn(sess)

    print('finish building graph')
    img_tlt = lines.size
    prc_g = []
    for gpu in range(nb_gpus):
        img_per_gpu = img_tlt - img_tlt / nb_gpus if i == 0 else img_tlt / nb_gpus
        prc_g.append(Process(target=eval, args=(gpu, img_q, sess, subgraph[gpu], img_per_gpu)))
        prc_g[-1].daemon=True
        prc_g[-1].start()


for i in range(num_t):
    prc_l[i].join()
for gpu in range(nb_gpus):
    prc_g[gpu].join()
