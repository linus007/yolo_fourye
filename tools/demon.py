import _init_paths

from net.yolo_tiny_net import YoloTinyNet
from detect.detect import detector
import os
import argparse
import cv2
from utils.timer import Timer
from utils.process_config import process_config
from datasets.pascal_voc import pascal_voc
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight_file"
        , default="weight_dir/save.ckpt-0", type=str)
    parser.add_argument("--img", type=str)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--cfg"
        , default="../config/train.cfg", type=str)
    parser.add_argument("--is_img", default="1" , type=int)
    args = parser.parse_args()

    data_conf, common_conf, net_conf, solver_conf   \
        = process_config(args.cfg)
    conf = {'phase': 'test'}
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    imdb = pascal_voc(data_conf, common_conf, conf)

    yolo = YoloTinyNet(common_conf, net_conf, False)
    detect = detector(yolo, imdb, args.weight_file
        , common_conf, net_conf)
    print(type(args.img))
    if args.is_img:
        assert os.path.exists(args.img)
        detect.image_detect(args.img)
    else:
        cap = cv2.VideoCapture(-1)
        detect.camera_dtector(cap)

if __name__ == '__main__':
    main()
