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

from nets.resnet_v1 import resnetv1

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ElementTree
from xml.dom import minidom

from final.config import CLASSES

IMAGES_DIR = os.path.join('labeling/JPEGImages')
OUTPUT_DIR = os.path.join('labeling/Annotations')

def create_xml(image_name, shape):
    height, width, channels = shape

    xml_annotation = ET.Element('annotation')
    xml_folder = ET.SubElement(xml_annotation, 'folder')
    xml_folder.text = 'JPEGImages'
    xml_filename = ET.SubElement(xml_annotation, 'filename')
    xml_filename.text = image_name
    xml_path = ET.SubElement(xml_annotation, 'path')
    xml_path.text = os.path.abspath(os.path.join(os.getcwd(), IMAGES_DIR, image_name))
    
    xml_source = ET.SubElement(xml_annotation, 'source')
    xml_database = ET.SubElement(xml_source, 'database')
    xml_database.text = 'Unknown'

    # Configure Size
    xml_size = ET.SubElement(xml_annotation, 'size')
    xml_width = ET.SubElement(xml_size, 'width')
    xml_width.text = str(width)
    xml_height = ET.SubElement(xml_size, 'height')
    xml_height.text = str(height)
    xml_depth = ET.SubElement(xml_size, 'depth')
    xml_depth.text = str(channels)

    xml_segmented = ET.SubElement(xml_annotation, 'segmented')
    xml_segmented.text = '0'

    return xml_annotation

def mark_object(xml_element, class_name, box, shape):
    height, width, channels = shape

    xmin = int(box[0]) + 1
    ymin = int(box[1]) + 1
    xmax = int(box[2]) + 1
    ymax = int(box[3]) + 1
    truncated = '1' if xmin == 1 or ymin == 1 or xmax == width or ymax == height else '0'
    difficult = '0'

    xml_object = ET.SubElement(xml_element, 'object')
    xml_name = ET.SubElement(xml_object, 'name')
    xml_name.text = class_name
    xml_pose = ET.SubElement(xml_object, 'pose')
    xml_pose.text = 'Unspecified'
    xml_truncated = ET.SubElement(xml_object, 'truncated')
    xml_truncated.text = truncated
    xml_difficult = ET.SubElement(xml_object, 'difficult')
    xml_difficult.text = difficult
    xml_box = ET.SubElement(xml_object, 'bndbox')
    xml_xmin = ET.SubElement(xml_box, 'xmin')
    xml_xmin.text = str(xmin)
    xml_ymin = ET.SubElement(xml_box, 'ymin')
    xml_ymin.text = str(ymin)
    xml_xmax = ET.SubElement(xml_box, 'xmax')
    xml_xmax.text = str(xmax)
    xml_ymax = ET.SubElement(xml_box, 'ymax')
    xml_ymax.text = str(ymax)
    return xml_object


def label(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the input image
    im_file = os.path.join(IMAGES_DIR, image_name)
    assert os.path.exists(im_file), "Image does not exist: {}".format(im_file)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.01

    xml = create_xml(image_name, im.shape)

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
            
            mark_object(xml, cls, bbox, im.shape)

    minidom_xml = minidom.parseString(ET.tostring(xml))
    annotation_filename = os.path.splitext(image_name)[0]+'.xml'
    annotation_path = os.path.join(OUTPUT_DIR, annotation_filename)
    with open(annotation_path, "w") as xml_file:
        minidom_xml.writexml(xml_file, addindent='\t', newl='\n')

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.USE_GPU_NMS =False

    # model path
    tfmodel = os.path.join('output', 'res101', 'voc_final_trainval', 'default', 'res101_faster_rcnn_iter_1000.ckpt')

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

    for root, dirs, files in os.walk(IMAGES_DIR):  
        for filename in files:
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Labeling for {}/{}'.format(IMAGES_DIR, filename))
            label(sess, net, filename)
