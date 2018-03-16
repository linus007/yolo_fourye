import tensorflow as tf
import numpy as np
import os
import cv2
import argparse
from net.yolo_tiny_net import YoloTinyNet
from utils.timer import Timer
from utils.nms import nms

import config.cfg as cfg

class detector(object):
    def __init__(self, net, imdb, weight_file
        , common_conf, net_conf):
        self._net = net
        self._weights_file = weight_file

        self._imdb = imdb
        self._classes = imdb.classes
        self._num_classes = len(self._classes)

        print(self._num_classes)
        self._image_size = int(common_conf['image_size'])
        self._cell_size = int(common_conf['cell_size'])
        self._boxes_per_cell = int(net_conf['boxes_per_cell'])
        self._thresh_hold = cfg.THRESHOLD
        self._iou_thresh_hold = cfg.IOU_THRESHOLD

        self._boundary1 = self._cell_size * self._cell_size \
            * self._num_classes
        self._boundary2 = self._boundary1   \
            + self._cell_size * self._cell_size * self._boxes_per_cell

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

        print("Restoring weights from: " + self._weights_file)
        self._saver = tf.train.Saver(tf.global_variables())
        self._saver.restore(self._sess, self._weights_file)


    def image_detect(self, img_path, wait=0):
        detect_timer = Timer()
        img = cv2.imread(img_path)

        detect_timer.tic()
        res = self.detect(img)
        detect_timer.toc()
        print('Average detecting time: {:.3f}'  \
            .format(detect_timer.average_time))
        self.draw_result(img, res)
        cv2.imshow('image', img)
        cv2.waitKey(wait)

    def detect(self, img):
        img_h, img_w, _ = img.shape
        inp = cv2.resize(img, (self._image_size, self._image_size))

        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)    \
            .astype(np.float32)
        inp = (inp / 255.0) * 2.0 - 1.0
        inp = np.reshape(inp,
            (1, self._image_size, self._image_size, 3))

        res = self.detect_from_cvmat(img_w, img_h, inp)
        return res

    def draw_result(self, img, res):
        for cls, boxes in res.items():
            for box in boxes:
                x1 =  int(box[0])
                y1 =  int(box[1])
                x2 =  int(box[2])
                y2 =  int(box[3])
                score = box[4]
                cv2.rectangle(img, (x1, y1)
                    , (x2, y2), (0, 255, 0), 2)

                cv2.putText(img, self._classes[int(cls)] + ": %.2f"%score
                    , (max(x1 + 5, 1), max(y1 - 7, 1)), cv2.FONT_HERSHEY_SIMPLEX
                    , 0.5, (0, 0, 0), 1, cv2.CV_AA)
    def camera_dtector(self, cap, wait=10):
        detect_timer = Timer()
        ret, _ = cap.read()
        while ret:
            ret, frame = cap.read()
            detect_timer.tic()
            res = self.detect(frame)
            detect_timer.toc()
            print('Average detecting time:{:.3f}s'.format(detect_timer.average_time))
            self.draw_result(frame, res)
            cv2.imshow("camera", frame)
            cv2.waitKey(wait)
            ret, frame = cap.read()

    def detect_from_cvmat(self, img_w, img_h, inputs):
        net_output = self._sess.run(self._net.logits
            , feed_dict={self._net._images:inputs})
        res = self.interpret_output(img_w, img_h, net_output)
        return res

    def coordinate_transfer(self, img_w, img_h, boxes):
        x1 = boxes[:, :, :, 0] - boxes[:, :, :, 2] / 2
        y1 = boxes[:, :, :, 1] - boxes[:, :, :, 3] / 2
        x2 = boxes[:, :, :, 0] + boxes[:, :, :, 2] / 2
        y2 = boxes[:, :, :, 1] + boxes[:, :, :, 3] / 2

        boxes[:, :, :, 0] = 1.0 * x1 * img_w / self._image_size
        boxes[:, :, :, 1] = 1.0 * y1 * img_h / self._image_size
        boxes[:, :, :, 2] = 1.0 * x2 * img_w / self._image_size
        boxes[:, :, :, 3] = 1.0 * y2 * img_h / self._image_size

        return boxes


    def interpret_output(self, img_w, img_h, output):
        # NOTE: duplicate code here

        output = np.reshape(output, output.shape[-1])
        class_prob = output[0:self._boundary1]
        class_prob = np.reshape(class_prob  \
            , [self._cell_size, self._cell_size, self._num_classes])

        scales = output[self._boundary1:self._boundary2]
        scales = np.reshape(scales,
            [self._cell_size, self._cell_size, self._boxes_per_cell])

        boxes = output[self._boundary2:]
        boxes = np.reshape(boxes,
            [self._cell_size, self._cell_size, self._boxes_per_cell, 4])

        offset = [np.arange(self._cell_size)]     \
            * self._cell_size * self._boxes_per_cell
        offset = np.reshape(offset,     \
            [self._boxes_per_cell, self._cell_size, self._cell_size])
        offset = np.transpose(offset, [1, 2, 0])

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] = 1.0 * boxes[:, :, :, :2] / self._cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])
        # duplicate code here
        boxes *= self._image_size
        self.coordinate_transfer(img_w, img_h, boxes)

        probs = np.zeros((self._cell_size, self._cell_size
            , self._boxes_per_cell, self._num_classes))

        for i in range(self._boxes_per_cell):
            for j in range(self._num_classes):
                tmp = scales[:, :, i] * class_prob[:, :, j]
                probs[:, :, i, j] = tmp * (tmp >= cfg.THRESHOLD)


        probs = np.transpose(probs, (3, 0, 1, 2))
        probs = np.reshape(probs, (self._num_classes
            , self._cell_size * self._cell_size     \
                * self._boxes_per_cell))
        boxes = np.reshape(boxes
            , (self._cell_size * self._cell_size * self._boxes_per_cell
                , 4))
        res = {}
        for i in range(len(self._classes)):
            prob = np.reshape(probs[i]
                , [self._cell_size * self._cell_size * self._boxes_per_cell
                    , 1])
            dets = np.hstack([boxes, prob])
            keep_inds = nms(dets, self._iou_thresh_hold)
            if len(keep_inds) > 0:
                res[str(i)] = dets[keep_inds]
        return res
