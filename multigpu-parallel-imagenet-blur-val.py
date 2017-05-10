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
EvalRes = namedtuple('EvalRes', 'c1 c5 img_cnt')

slim = tf.contrib.slim

height, width = 299, 299
num_classes = 1001
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
import Queue as Q
from multiprocessing import Process, Lock, Event
from multiprocessing.queues import SimpleQueue, Queue
from math import floor
fetch_test = False
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
        processed_image = imresize(im, (width, height))
        processed_image = (processed_image.astype(np.float32) / 256 - 0.5) * 2
        processed_images = np.expand_dims(processed_image, axis=0)
        img_q.put([processed_images, label])

img_q = Queue()
num_t = int(sys.argv[2])
prc_l = []
for i in range(num_t):
    print('fetch thread %d' % i)
    prc_l.append(Process(target=fetch_img, args=(i, num_t, img_q)))
    prc_l[-1].daemon=True
    prc_l[-1].start()

import threading
counter = 0
counter_lock = threading.Lock()

def eval(gpu, img_q, res_q, end_eval):
    ###############
    # Build graph #
    ###############
    with tf.Graph().as_default():
        print('loading graph on /gpu:%d' % gpu)
        with tf.name_scope('tower-%d' % gpu):
            eval_input = tf.placeholder(tf.float32, [1, height, width, 3])
            with slim.arg_scope(inception_arg_scope()):
                with counter_lock:
                    global counter
                    resue = None if counter == 0 else True
                    counter += 1
                logits, _ = inception_v4(eval_input, num_classes,
                                         reuse=resue, scope='InceptionV4',
                                         is_training=False)
            predictions = tf.argmax(logits, 1)
            probabilities = tf.nn.softmax(logits)
        
        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = str(gpu)
        config.gpu_options.allow_growth = True
        # config.log_device_placement = True
        # config.allow_soft_placement = True
        sess = tf.Session(config=config)

        init_fn = slim.assign_from_checkpoint_fn(
            'checkpoints/inception_v4.ckpt',
            slim.get_model_variables('InceptionV4'))
        init_fn(sess)

    ##############
    # Eval image #
    ##############
    st = time.time()
    img_cnt = 0
    c1, c5 = 0, 0
    while not end_eval.is_set():
        try:
            processed_images, label = img_q.get(False)
        except Q.Empty:
            continue
        if fetch_test:
            img_cnt += 1
            sys.stdout.write('\r{:07d}/{:07d}'.format(img_cnt, img_tlt))
            sys.stdout.flush()
            continue
        pred, prob = sess.run([predictions, probabilities], feed_dict={
                eval_input: processed_images
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
        print('gpu %d images: %d\ttop 1: %0.4f\ttop 5: %0.4f' % (gpu, idx + 1, c1/(idx + 1), c5/(idx + 1)))
        img_cnt += 1
    res_q.put(EvalRes(c1, c5, img_cnt))
    print('gpu %d finish in %.4f sec' % (gpu, time.time()-st))

print('begin building graph')
img_tlt = lines.size
prc_g = []
res_q = SimpleQueue()
end_eval = Event()
for gpu in range(nb_gpus):
    prc_g.append(Process(target=eval,
                         args=(gpu, img_q, res_q, end_eval)))
    prc_g[-1].daemon=True
    prc_g[-1].start()

for i in range(num_t):
    prc_l[i].join()
    print('join prefetch thread %d' % i)
print('prefetch end')

end_eval.set()
print('end eval')

eval_cnt = 0
c1, c5 = 0, 0
for gpu in range(nb_gpus):
    print('join gpu %d' % gpu)
    prc_g[gpu].join()
    eval_res = res_q.get()
    eval_cnt += eval_res.img_cnt
    c1       += eval_res.c1
    c5       += eval_res.c5
assert eval_cnt == img_tlt
print('='*40)
print('top 1: %0.4f\ttop 5: %0.4f' % (c1/img_tlt, c5/img_tlt))
print('='*40)

