#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import numpy as np
import os, cv2

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

from final.config import CLASSES

DEMO_IMAGES_DIR = os.path.join('data/final/demo')

FONT = cv2.FONT_HERSHEY_SIMPLEX


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(DEMO_IMAGES_DIR, image_name)
    assert os.path.exists(im_file), "Image does not exist: {}".format(im_file)
    im = cv2.imread(im_file)

    # cv2.imshow('image_name', im)
    # cv2.waitKey(0)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.01
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            continue

        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]

            pt1 = (bbox[0], bbox[1])
            pt2 = (bbox[2], bbox[3])
            pt3 = (int(bbox[0] - 2), int(bbox[1] - 2))
            
            cv2.rectangle(im, pt1, pt2, (64, 64, 255), 2)
            cv2.putText(im, cls, pt1, FONT, 1, (0, 0, 64), 2, cv2.LINE_AA)
            cv2.putText(im, cls, pt3, FONT, 1, (32, 32, 186), 2, cv2.LINE_AA)

    cv2.destroyAllWindows()
    cv2.imshow('image_name', im)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.USE_GPU_NMS =False

    # model path
    tfmodel = os.path.join('output', 'res101', 'voc_final_trainval', 'default', 'res101_faster_rcnn_iter_5000.ckpt')

    print(tfmodel)							  
    if not os.path.isfile(tfmodel + '.meta'):
    
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    net = resnetv1(num_layers=101)
    net.create_architecture("TEST", len(CLASSES),
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))

    for root, dirs, files in os.walk(DEMO_IMAGES_DIR):  
        for filename in files:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Demo for {}/{}'.format(DEMO_IMAGES_DIR, filename))
            demo(sess, net, filename)

    plt.show()
