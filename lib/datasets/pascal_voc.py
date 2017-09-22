from __future__ import print_function
import os
import os.path as osp
import numpy as np
import xml.etree.ElementTree as ET
from datasets.imdb import imdb
import cPickle

DEBUG = False
class pascal_voc(imdb):
    def __init__(self, data_conf, common_conf, conf):
        imdb.__init__(self, data_conf,common_conf, conf)
        self._classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._class_to_ind = dict(  \
            zip(self.classes, range(self.num_classes)))
        self._image_set = data_conf['image_set']
        self._image_index = self._load_image_set_idex()
        self._all_image_paths = self.all_img_paths()
        self._cache_file = self.cache_file()
        self.debug()

    def get_gt_infos(self):
        """
        get grount truth information
        Return:
            gt_infos = {'image_paths': image_paths
                , gt_boxes, 'classes': classes}
            gt_boxes = [ing_inds x obj_inds x [x1, y1, x2, y2]]
        """
        if os.path.exists(self._cache_file):
            with open(self._cache_file) as f:
                gt_infos = cPickle.load(f)
                print("{} gt info loaded from {}"   \
                    .format(self.name, self._cache_file))
                return gt_infos

        gt_boxes = []
        gt_classes = []
        for inds in  self._image_index:
            box, cls = self._load_annotation(inds)
            gt_boxes.append(box)
            gt_classes.append(cls)

        gt_infos = {'image_paths': self._all_image_paths
                , 'boxes':gt_boxes, 'classes':gt_classes}

        with open(self._cache_file, "wb") as f:
            print("writing gt info to %s..."%self._cache_file)
            cPickle.dump(gt_infos, f, cPickle.HIGHEST_PROTOCOL)
            print("writing over")

        return gt_infos



    def _load_image_set_idex(self):
        """
        load the indexes listed in this dataset's image set file.
        """

        image_set_file = osp.join("..", self.data_path
            , "VOCdevkit2007/VOC2007/ImageSets/Main"
            , self._image_set + ".txt")

        assert os.path.exists(image_set_file)   \
            , "image set: %s not existed"%image_set_file

        with open(image_set_file) as f:
            image_index = [inds.strip() for inds in f.readlines()]

        return image_index


    def cache_file(self):
        return os.path.join(self.cache_dir
            , self.name + "_" + self._image_set + ".pkl")


    def all_img_paths(self):
        return [osp.join("..", self.data_path
                , "VOCdevkit2007/VOC2007/JPEGImages"
                , inds + ".jpg")
            for inds in self._image_index]


    def  _load_annotation(self, index):
        """
        Load image and bounding boxes and its labels from XML file in
        the PASCAL VOCmdb = pascal_voc("PASCAL_VOC", "trainval", data_conf, common_conf, conf)
        Arg:
            index: the index of target image(or XML)
        Return:
            gt_info:[min_x, min_y, max_x, max_y, class]
            classes: [class]
        """
        file_name = os.path.join("..", self.data_path
            , "VOCdevkit2007/VOC2007/Annotations", index +".xml")
        assert os.path.exists(file_name), "%s not existed"%file_name

        tree = ET.parse(file_name)
        objs = tree.findall('object')

        num_objs = len(objs)

        gt_boxes = np.zeros((num_objs, 4))
        gt_classes = np.zeros((num_objs), np.int32)

        for ix, obj in enumerate(objs):
            bbox = obj.find("bndbox")
            # make pixel indexes 0 based
            x1 = float(bbox.find("xmin").text) - 1
            y1 = float(bbox.find("ymin").text) - 1
            x2 = float(bbox.find("xmax").text) - 1
            y2 = float(bbox.find("ymax").text) - 1

            cls = self._class_to_ind[
                obj.find('name').text.lower().strip()]

            gt_boxes[ix,:] = [x1, y1, x2, y2]
            gt_classes[ix] = cls

        return gt_boxes, gt_classes


    def debug(self):
        if DEBUG:
            # all iamge paths debug
            print("all image paths debug:")
            print("length of image paths:%d"%len(self._all_image_paths))
            print("the path of first 10 image paths:")
            print(self._all_image_paths[:10])
            print("all image paths debug done")
            print("***************************************")
            print("")

            # image set index debug
            print("image set index debug:")
            print("length of image set: %d"%len(self._image_index))
            print("index of first 10 image set:")
            print(self._image_index[:10])
            print("index of last 10 image set:")
            print(self._image_index[-10:])
            print("image set index debug done.")
            print("***************************************")
            print("")


            # cache file
            print("cache file:")
            print(self._cache_file)
            print("***************************************")
            print("")

            # gt info
            gt_infos = self.get_gt_infos()
            img_paths = gt_infos["image_paths"]
            boxes = gt_infos["boxes"]
            clses = gt_infos["classes"]
            assert(len(img_paths) == len(boxes) == len(clses))
            print("length of imagess:%d"%len(img_paths))
            print("paths of first 10 images:")
            print(img_paths[:10])
            print("boxes of first 10 images:")
            print(boxes[:10])
            print("classes of first 10 images:")
            print(clses[:10])
            print("***************************************")
            print("")

            """
            # gt info process
            import cv2
            imgs, labs = self.gt_info_process(gt_infos)
            print("length of images:%d"%len(imgs))
            print("shape{}".format(imgs[0].shape))
            print(img_paths[-1])
            print("imgpath:{}".format(img_paths[0]))
            img_0_path = osp.join(self.cache_dir, img_paths[0].split("/")[-1])
            img_1_path = osp.join(self.cache_dir, img_paths[-1].split("/")[-1])
            print(img_0_path)
            cv2.imwrite(img_0_path, imgs[0])
            cv2.imwrite(img_1_path, imgs[-1])

            print("lables:")
            print("is_obj")
            print(labs[-1, :, :, 0])
            print("box_info:")
            print(labs[-1, :, :, 1:5])
            print("clses:")
            print(labs[-1, :, :, 5:] > 0)
            """
            imgs, labs = self.prepare()
            print("length of img_paths:{}".format(len(imgs)))
            print("print the last 10 images:")
            print(imgs[-10:])
            print("lables:")
            print("is_obj")
            print(labs[-1, :, :, 0])
            print("box_info:")
            print(labs[-1, :, :, 1:5])
            print("clses:")
            print(labs[-1, :, :, 5:] > 0)


            print("*************************************")

            inds = np.arange(5000)
            inds = np.random.choice(inds, size=10)
            print("inds: ")
            print(inds)
            print('image paths:')
            img_paths = np.array(self.image_paths)
            print(img_paths[inds])
            imgs, lables = self.get(inds)
            print('imgs:')
            print(imgs)
            print('objs:')
            print(lables[:, :, :, 0])
            print('info:')
            print(lables[:, :, :, 1:5])

            print('classes:')
            print(lables[:, :, :, 5:])
