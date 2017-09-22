from __future__ import absolute_import
from __future__ import print_function
import _init_paths

import argparse
import os

from utils.process_config import process_config
from utils.data_batch import databatch

from datasets.pascal_voc import pascal_voc
from net.yolo_tiny_net import YoloTinyNet
from solver.yolo_tiny_solver import YoloSolver

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', default=0.2, type=float)
    parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--cfg_file', default='../config/train.cfg'  \
        , type=str)
    conf = {'phase': 'train'}

    args = parser.parse_args()
    cfg_file = args.cfg_file
    data_conf, common_conf, net_conf, solver_conf   \
        = process_config(cfg_file)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    imdb = pascal_voc(data_conf, common_conf, conf)
    imdb.prepare()
    print('preparing data batch...')
    data_batch = databatch(int(common_conf['batch_size'])
        , int(data_conf['cache_size']), imdb)
    data_batch.ready()
    print('data batch prepared done')

    print("initializing net...")
    yolo_net = YoloTinyNet(common_conf, net_conf)
    print("net initializing done")
    print("initializing solver...")
    solver = YoloSolver(data_batch, yolo_net
        , common_conf, solver_conf, data_conf)
    print("solver initializing done")

    print("start training...")
    solver.train()
    print('Done training...')

if __name__ == '__main__':
    main()
