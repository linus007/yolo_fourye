import os
import os.path as osp
import cv2
import numpy as np
import cPickle

class imdb(object):
    def __init__(self, data_conf, common_conf, conf):
        self._name = data_conf['name']
        self._num_classes = 0
        self._classes = []
        self._conf = data_conf
        self._data_path = data_conf.get("data_path")
        self._image_size = int(common_conf.get("image_size"))
        self._cell_size = int(common_conf.get("cell_size"))
        self._batch_size = int(common_conf['batch_size'])
        self._phase = conf.get("phase")
        self._flipped = bool(int(data_conf.get("flipped")))
        self._images = None
        self._lables = None
        self._eps = common_conf.get("eps")
        self._cache_size = int(data_conf['cache_size'])
        self.epoch = 1
        self.has_flipped = False
        self.flipped_tag = None

    @property
    def name(self):
        return self._name;

    @property
    def classes(self):
        return self._classes

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def data_path(self):
        return self._data_path

    @property
    def cache_file(self):
        raise NotImplementedError

    @property
    def all_img_paths(self):
        raise NotImplementedError

    def get_gt_infos(self):
        raise NotImplementedError

    @property
    def cache_dir(self):
        cache_dir = os.path.join("..", self.data_path, "cache")
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        return cache_dir

    @property
    def image_size(self):
        if self._image_paths is None:
            return 0
        return len(self._image_paths)

    @property
    def image_paths(self):
        if self._image_paths is None:
            self.prepare()
        return self._image_paths

    @property
    def lables(self):
        if self._lables is None:
            self.prepare()
        return self._lables

    def get(self, inds):
        image_paths = np.array(self._image_paths)
        img_paths = image_paths[inds]
        imgs, _, _ = self.readImages(img_paths)
        lables = self._lables[inds]

        return imgs, lables


    def readImages(self, img_paths, flipped_tags=None):

        if flipped_tags != None:
            assert len(img_paths) == len(flipped_tags)

        imgs = np.zeros(
            shape=(len(img_paths), self._image_size
                , self._image_size, 3))
        wts = []
        hts = []
        for inds, img_path in enumerate(img_paths):
            assert osp.exists(img_path) \
                , "image path: %s not existed"%img_path

            img, wt, ht = self.readImage(img_path=img_path)
            hts.append(ht)
            wts.append(wt)
            if flipped_tags != None and flipped_tags[inds]:
                img = img[:, ::-1, :]
            imgs[inds, :, :, :] = img
        return imgs, wts, hts

    def readImage(self, img_path):
        assert osp.exists(img_path) \
            , "image path: %s not existed"%img_path
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        ht = img.shape[0]
        wt = img.shape[1]
        img = cv2.resize(img, (self._image_size, self._image_size))
        img = img / 255.0 * 2.0 - 1
        #img = img / 255.0

        return img, wt, ht


    def add_flipped(self, image_paths, lables):
        print("add horizontally-flipped training examples...")

        self.has_flipped = True
        fl_lables = lables.copy()[:, :, ::-1, :]
        fl_lables[:, :, :, 1] = (self._image_size   \
                - 1 - fl_lables[:, :, :, 1])    \
            * fl_lables[:, :, :, 0]
        print("flipped done")
        image_paths *= 2
        lables = np.vstack([lables, fl_lables])

        return image_paths, lables;

    def load_image_paths_lables(self):
        cache_file = osp.join(self.cache_dir
            , self.name + "_" + self._phase + "gt_lables.pkl")

        if osp.exists(cache_file):
            print("loading gt_lables from: {}".format(cache_file))
            with open(cache_file, "rb") as f:
                image_paths, lables = cPickle.load(f)
                print("loading done")
                return image_paths, lables

        print("getting gt_infos...")
        gt_infos = self.get_gt_infos()
        print("getting gt_infos done")
        print("process gt_infos...")
        image_paths, lables = self.gt_info_process(gt_infos)
        print("infos processing done")

        print("save images and lables into the {}".format(cache_file))
        with open(cache_file, "wb") as f:
            cPickle.dump([image_paths, lables], f, cPickle.HIGHEST_PROTOCOL)
        return image_paths, lables




    def prepare(self):
        print("preparing images and its lables")
        image_paths, lables = self.load_image_paths_lables()
        flipped_tag = np.zeros(len(image_paths))
        if self._flipped and not self.has_flipped:
            image_paths, lables = self.add_flipped(image_paths, lables)
            flipped_tag = np.hstack([flipped_tag
                , np.ones(len(flipped_tag))])
        print("preparing done")
        self._lables = lables
        self._image_paths = image_paths
        self.flipped_tag = flipped_tag
        return self._image_paths, self._lables


    def gt_info_process(self, gt_infos):
        #gt_infos = {'image_paths': self._all_image_paths
        #        , 'boxes':gt_boxes, 'classes':gt_classes}
        img_paths = gt_infos["image_paths"]
        boxes = gt_infos["boxes"]
        classes = gt_infos["classes"]

        lables = np.zeros((len(boxes)
            , self._cell_size, self._cell_size, 25))

        for inds in range(len(boxes)):
            _, wt, ht = self.readImage(img_paths[inds])
            h_ration = float(self._image_size) / (ht + 1e-14)
            w_ration = float(self._image_size) / (wt + 1e-14)

            for box, cls in zip(boxes[inds], classes[inds]):
                x1, y1, x2, y2 = box
                x_ctr = (x1 + x2) * w_ration / 2.0
                y_ctr = (y1 + y2) * h_ration / 2.0
                w_box = (x2 - x1) * w_ration
                h_box = (y2 - y1) * h_ration



                x_inds = int(x_ctr * self._cell_size / self._image_size)
                y_inds = int(y_ctr * self._cell_size / self._image_size)

                assert (x_inds <= 7   \
                    or x_inds >= 0 or y_inds >= 0     \
                    or y_inds <= 7)    \
                    , "inds:{}, x_inds:{}, y_inds:{}".format(inds, x_inds, y_inds)

                if lables[inds, y_inds, x_inds, 0] == 1:
                    continue

                lables[inds, y_inds, x_inds, 0] = 1
                lables[inds, y_inds, x_inds, 1:5] = [x_ctr, y_ctr, w_box, h_box]
                lables[inds, y_inds, x_inds, 5 + cls] = 1

        return img_paths, lables


    def debug(self):
        raise NotImplementedError
