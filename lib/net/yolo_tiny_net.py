from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from net.net import Net
class YoloTinyNet(Net):
    def __init__(self, common_conf, net_conf, is_training=True):
        """
        common_conf: a params dict
        net_conf: a params dict
        """
        Net.__init__(self, net_conf)
        self._image_size = int(common_conf['image_size'])
        self._num_classes = int(common_conf['num_classes'])
        self._batch_size = int(common_conf['batch_size'])

        self._cell_size = int(common_conf['cell_size'])
        self._boxes_per_cell = int(net_conf['boxes_per_cell'])
        self._class_scale = float(net_conf['class_scale'])
        self._coord_scale = float(net_conf['coord_scale'])
        self._objcet_scale = float(net_conf['object_scale'])
        self._noobject_scale = float(net_conf['noobject_scale'])

        self._boundary1 = self._cell_size * self._cell_size * self._num_classes
        self._boundary2 = self._boundary1    \
            + self._cell_size * self._cell_size * self._boxes_per_cell

        self._offset = np.transpose(np.reshape( \
                np.array(   \
                    [np.arange(self._cell_size)]    \
                    * self._cell_size * self._boxes_per_cell)   \
                , (self._boxes_per_cell, self._cell_size, self._cell_size)) \
            , (1, 2, 0))

        self._images = tf.placeholder(tf.float32
            , [None, self._image_size, self._image_size, 3]
            , name='images')

        self.logits = self.inference(self._images)

        if is_training:
            self._lables = tf.placeholder(tf.float32
                , [None, self._cell_size, self._cell_size   \
                    , 5 + self._num_classes])
            print("num_classes:", self._lables.get_shape())

            self.loss_layer(self.logits, self._lables)
            self.total_loss = tf.losses.get_total_loss()
            print(type(self.total_loss))
            print("total_loss:")
            print(self.total_loss.get_shape())
            tf.summary.scalar("total_loss", self.total_loss)



    def inference(self, inputs):
        """
        Build the yolo model
        Args:
            inputs(images): 4-D tensor [b_num_classesatch_size, image_height, image_width, channel]
        Return:
            4-D tensor [batch_size, cell_size, cell_size
            self.logits = self.inference(self._images)    , num_classes + 5 * boxces_per_cell]
        """

        """
        def conv(self, intput, k_h, k_w, c_o, s_h, s_w
            , name, activiation=self.leaky_relu, padding=DEFAULT_PADDING
            , group=1, trainable=True):
        def max_pool(self, input, k_h, k_w, s_h, s_w
            , name, padding=DEFAULT_PADDING):
        """

        conv1 = self.conv(inputs, 3, 3, 16, 1, 1
            , "conv1")


        pool1 = self.max_pool(conv1, 2, 2, 2, 2
            , "pool1")


        conv2 = self.conv(pool1, 3, 3, 32, 1, 1
            , "conv2")
        tf.summary.histogram("conv2",conv2)
        pool2 = self.max_pool(conv2, 2, 2, 2, 2
            , "pool2")

        conv3 = self.conv(pool2, 3, 3, 64, 1, 1
            , "conv3")
        pool3 = self.max_pool(conv3, 2, 2, 2, 2
            , "pool3")

        conv4 = self.conv(pool3, 3, 3, 128, 1, 1
            , "conv4")
        pool4 = self.max_pool(conv4, 2, 2, 2, 2
            , "pool4")

        conv5 = self.conv(pool4, 3, 3, 256, 1, 1
            , "conv5")
        pool5 = self.max_pool(conv5, 2, 2, 2, 2
            , "pool5")

        conv6 = self.conv(pool5, 3, 3, 512, 1, 1
            , "conv6")
        pool6 = self.max_pool(conv6, 2, 2, 2, 2
            , "pool6")

        conv7 = self.conv(pool6, 3, 3, 1024, 1, 1
            , "conv7")

        conv8 = self.conv(conv7, 3, 3, 1024, 1, 1
            , "conv8")

        conv9 = self.conv(conv8, 3, 3, 1024, 1, 1
            , "conv9")

        transpose = tf.transpose(conv9, (0, 3, 1, 2))
        """
        def fc(self, name, input, num_in, num_out
            , activiation=self.leaky_relu, trainable=True):
        """
        # Fully connected layer
        fc10 = self.fc("local1", transpose
            , self._cell_size * self._cell_size * 1024, 256)

        fc11 = self.fc("local2", fc10, 256,  4096)

        fc12 = self.fc("local3", fc11, 4096   \
            , self._cell_size * self._cell_size \
                * (self._num_classes + self._boxes_per_cell * 5)   \
            , is_leaky=False
            , pretrainable=False)

        """
        n1 = self._cell_size * self._cell_size * self._num_classes
        n2 = n1 + self._cell_size * self._cell_size * self._boxes_per_cell

        class_probs = tf.reshape(fc12[:, 0:n1]
            , (-1, self._cell_size, self._cell_size, self._num_classes))
        scales = tf.reshape(fc12[:, n1:n2]
            , (-1, self._cell_size, self._cell_size, self._boxes_per_cell))
        boxes = tf.reshape(fc12[:, n2:]
            , (-1, self._cell_size, self._cell_size, self._boxes_per_cell * 4))

        fc12 = tf.concat([class_probs, scales, boxes], 3)
        print("fc12_shape:{}".format(fc12.get_shape()))
        predicts = fc12
        """
        return fc12

    def cal_iou(self, boxes1, boxes2, scope='iou'):
        """
        Calculate ious
        Args:
            boxes1: 5-D tensor [batch_size, cell_size, cell_size
                , boxes_per_cell, 4] ===> (x_center, y_center, w, h)
            boxes2: 5-D tensor [batch_size, cell_size, cell_size,
                , 1, 4] ===> (x_center, y_center, w, h)
        Return:
            iou: 4-D tensor [batch_size, cell_size, cell_size, boxes_per_cell]
        """
        with tf.variable_scope(scope):
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2
                             , boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2
                             , boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2
                             , boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2])
            boxes1 = tf.transpose(boxes1, (1, 2, 3, 4, 0))

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2
                             , boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2
                             , boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2
                             , boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2])
            boxes2 = tf.transpose(boxes2, (1, 2, 3, 4, 0))


            # calculate the left up point
            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
            # calculate the right bottom point
            rb = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

            # intersection
            intersection = tf.maximum(0.0, rb - lu)
            inter_squrare = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            # calculate the boxes1 square and box2 square
            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0])   \
                * (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0])   \
                * (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_squrare, 1e-14)

        return tf.clip_by_value(inter_squrare / union_square, 0.0, 1.0)


    def loss_layer(self, predicts, lables, scope='loss_layer'):
        print("predicts:{}".format(predicts.get_shape()))
        with tf.variable_scope(scope):

            predict_classes = tf.reshape(predicts[:, :self._boundary1]
                , [self._batch_size, self._cell_size
                    , self._cell_size, self._num_classes])

            predict_scales = tf.reshape(predicts[:, self._boundary1:self._boundary2]
                , [self._batch_size, self._cell_size
                    , self._cell_size, self._boxes_per_cell])



            predict_boxes = tf.reshape(predicts[:, self._boundary2:]
                , [self._batch_size, self._cell_size
                    , self._cell_size, self._boxes_per_cell, 4])

            response = tf.reshape(lables[:, :, :, 0]
                , [self._batch_size, self._cell_size
                    , self._cell_size, 1])

            boxes = tf.reshape(lables[:, :, :, 1:5]
                , [self._batch_size, self._cell_size
                    , self._cell_size, 1, 4])
            boxes = tf.tile(boxes
                , [1, 1, 1, self._boxes_per_cell, 1]) / self._image_size

            classes = lables[:, :, :, 5:]

            offset = tf.constant(self._offset, dtype=tf.float32)
            offset = tf.reshape(offset
                , [1, self._cell_size, self._cell_size, self._boxes_per_cell])
            offset = tf.tile(offset, [self._batch_size, 1, 1, 1])

            predict_boxes_tran = tf.stack(
                [(predict_boxes[:, :, :, :, 0] + offset) / self._cell_size  \
                , (predict_boxes[:, :, :, :, 1] \
                    + tf.transpose(offset, (0, 2, 1, 3))) / self._cell_size \
                , tf.square(predict_boxes[:, :, :, :, 2])   \
                , tf.square(predict_boxes[:, :, :, :, 3])])

            predict_boxes_tran = tf.transpose(predict_boxes_tran
                , (1, 2, 3, 4, 0))

            iou_predict_truth = self.cal_iou(predict_boxes_tran, boxes)

            # calculate I tensor [batch_size, cell_size, cell_size, boxes_per_cell]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast((iou_predict_truth >= object_mask)
                , dtype=tf.float32) * response

            # calculate no_I tensor [batch_size, cell_size, cell_size]
            noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) \
                - object_mask

            """
            boxes_tran = tf.stack([boxes[:, :, :, :, 0] \
                    * self._cell_size - offset \
                , boxes[:, :, :, :, 1] * self._cell_size    \
                    - tf.transpose(offset, (0, 2, 1, 3))    \
                , tf.sqrt(boxes[:, :, :, :, 2]) \
                , tf.sqrt(boxes[:, :, :, :, 3])])
            """
            boxes_tran = tf.stack([boxes[:, :, :, :, 0] \
                    - offset / self._cell_size  \
                , boxes[:, :, :, :, 1]     \
                    - tf.transpose(offset, (0, 2, 1, 3)) / self._cell_size    \
                , tf.sqrt(boxes[:, :, :, :, 2]) \
                , tf.sqrt(boxes[:, :, :, :, 3])])

            boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(class_delta)
                    , axis=[1, 2, 3])
                , name='class_loss') * self._class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(object_delta)
                    , axis=[1, 2, 3])
                , name='object_loss') * self._objcet_scale

            # noobjec1470t_loss
            # TODO: the same as the paper, but different from code from web
            # TODO: web: noobject_delta = noobject_mask * predict_scales
            noobject_delta = noobject_mask *(predict_scales - iou_predict_truth)
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(noobject_delta)
                    , axis=[1, 2, 3])
                , name='noobject_loss') * self._noobject_scale


            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(boxes_delta)
                    , axis=[1, 2, 3, 4])
                , name='coord_loss') * self._coord_scale


            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
            tf.summary.histogram('iou', iou_predict_truth)
