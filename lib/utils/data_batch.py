import numpy as np
from datasets import imdb
from concurrent_queue import ConcurrentQueue
import threading
class databatch(object):
    def __init__(self, batch_size, catch_size, imdb):

        self._batch_size = batch_size
        self._cache_size = catch_size
        self._cache = ConcurrentQueue(self._cache_size)

        self._imdb = imdb

        self._image_size = imdb.image_size
        self._image_inds = np.arange(imdb.image_size)
        self.epoch = 0


    def batch_producer(self):
        inds = np.random.choice(self._image_inds
            , self._batch_size, replace=False)

        imgs, lables = self._imdb.get(inds)
        assert len(imgs) == len(lables)
        return imgs, lables


    def add_one_batch(self):
        imgs, lables = self.batch_producer()

        self._cache.put([imgs, lables])


    def ready(self):
        print('ready to prepare the batch data')
        for i in range(self._cache_size):
            self._cache.put(self.batch_producer())

    def next_batch(self):
        imgs, lables = self._cache.get()

        thread = threading.Thread(target=self.add_one_batch)
        thread.start()
        self.epoch += 1
        return imgs, lables
