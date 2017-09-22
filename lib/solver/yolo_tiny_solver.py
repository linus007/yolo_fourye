from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re
import sys
import time
import datetime
import os
import os.path as osp

from solver.solver import Solver
from utils.timer import Timer

class YoloSolver(Solver):
    """
    Yolo Solver
    """
    def __init__(self, data_batch, net, common_conf, solver_conf, data_conf):
        # process params
        self._moment = float(solver_conf['moment'])
        self._batch_size = int(common_conf['batch_size'])


        self._data_path = str(data_conf['data_path'])
        self._output_dir = osp.join("..", self._data_path, 'output')
        if not osp.exists(self._output_dir):
            osp.makedirs(self._output_dir)

        self._pretrain_path = str(solver_conf['pretrain_model_path'])
        self._pretrain_path = osp.join("..", self._data_path
            , self._pretrain_path)
        #assert osp.exists(self._pretrain_path), self._pretrain_path

        self._train_dir = str(solver_conf['train_dir'])
        self._train_dir = osp.join("..", self._data_path
            , self._train_dir)
        if not osp.exists(self._train_dir):
            osp.makedirs(self._train_dir)

        self._max_iters = int(solver_conf['max_iterators'])
        self._save_iter = int(solver_conf['save_iter'])
        self._summary_iter = int(solver_conf['summary_iter'])

        self._learning_rate = float(solver_conf['learning_rate'])
        print("learning_rate:{}".format(self._learning_rate))
        self._decay_steps = int(solver_conf['decay_steps'])
        self._decay_rate = float(solver_conf['decay_rate'])
        self._stair_case = bool(solver_conf['stair_case'])

        self._num_classes = int(common_conf['num_classes'])

        self._ckpt_file = os.path.join(self._output_dir, 'save.ckpt')

        #
        self._data_batch = data_batch
        self._net = net

        # construct graph
        self.construct_graph()


    def construct_graph(self):

        self.variable_to_restore = tf.global_variables()
        self.saver = tf.train.Saver(self.variable_to_restore
            , max_to_keep=None)

        saver_load = tf.train.Saver()

        self.summary_op = tf.summary.merge_all()
        self._writer = tf.summary.FileWriter(self._output_dir, flush_secs=60)

        self.global_step = tf.get_variable( \
            'global_step', [], initializer=tf.constant_initializer(0)   \
            , trainable=True)

        self.learning_rate = tf.train.exponential_decay(
            self._learning_rate, self.global_step
            , self._decay_steps, self._decay_rate
            , self._stair_case, name="learning_rate")

        self.optimizer = tf.train.MomentumOptimizer(
            self.learning_rate, self._moment).minimize(
                self._net.total_loss)

        print('optimizer')
        self.ema = tf.train.ExponentialMovingAverage(decay=0.9999)
        self.averages_op = self.ema.apply(tf.trainable_variables())

        with tf.control_dependencies([self.optimizer]):
            self.train_op = tf.group(self.averages_op)

        gpu_options = tf.GPUOptions()
        config =  tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)


        self.sess.run(tf.global_variables_initializer())
        saver_load.restore(self.sess, self._pretrain_path)

        self._writer.add_graph(self.sess.graph)



    def train(self):

        train_timer = Timer()
        load_timer = Timer()

        for step in range(1, self._max_iters + 1):
            load_timer.tic()
            imgs, labs = self._data_batch.next_batch()
            load_timer.toc()

            feed_dict = {self._net._images: imgs, self._net._lables: labs}

            if step % self._summary_iter == 0:
                if step % (self._summary_iter * 10) == 0:
                    train_timer.tic()
                    summary_str, loss, _ = self.sess.run(
                        [self.summary_op
                            , self._net.total_loss
                            , self.train_op]
                        , feed_dict=feed_dict)
                    train_timer.toc()

                    log_str = ('{} Epoch: {}, Step: {}, Learning rate: {}'
                        ' Loss: {:5.3f}\nSpeed: {:.3f}s/iter,'
                        'Load: {:.3f}s/iter, Remain: {}').format(
                        datetime.datetime.now().strftime('%m/%d %H:%M:%s')
                        , self._data_batch.epoch
                        , int(step)
                        , round(self.learning_rate.eval(session=self.sess), 6)
                        , loss
                        , train_timer.average_time
                        , load_timer.average_time
                        , train_timer.remain(step, self._max_iters))
                    print(log_str)

                else:
                    train_timer.tic()
                    summary_str, _ = self.sess.run(
                        [self.summary_op, self.train_op]
                        , feed_dict=feed_dict)
                    train_timer.toc()

                self._writer.add_summary(summary_str, step)
            else:
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % self._save_iter == 0:
                print("{} Saving checkpoint file to: {}"
                    .format(datetime.datetime.now().strftime('%m/%d %H:%H:%S')
                        , self._output_dir))
                self.saver.save(self.sess, self._ckpt_file
                    , global_step=self.global_step)
