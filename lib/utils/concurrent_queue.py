import Queue
import threading

DEBUG = False
class ConcurrentQueue(object):
    def __init__(self, capacity=0):
        # init the capacity of Queue
        self.__capacity = capacity
        # init the mutex
        self.__mutex = threading.Lock()
        # init conditional variable
        self.__cond = threading.Condition(self.__mutex)
        # create queue
        self.__queue = Queue.Queue( \
            maxsize=self.__capacity)

    def put(self, elem):
        if self.__cond.acquire():
            """
            while self.__queue.qsize >= self.__capacity:
                self.__cond.wait()
            """
            if (self.__queue.qsize() < self.__capacity):
                self.__queue.put(elem)

            self.__cond.notify()
            self.__cond.release()

    def get(self):
        if self.__cond.acquire():
            if self.__queue.empty():
                elem = None
            else:
                elem = self.__queue.get()
            self.__cond.notify()
            self.__cond.release()

        return elem

    def clear(self):
        if self.__cond.acquire():
            self.__queue.queue.clear()
            self.__cond.release()
            self.__cond.notify_all()

    def empty(self):
        is_empty = False
        if self.__mutex.acquire():
            is_empty = self.__queue.empty()
            self.__mutex.release()

        return is_empty

    def size(self):
        size = 0
        if self.__mutex.acquire():
            size = self.__queue.qsize()
            self.__mutex.release()

        return size

def debug():
    q = ConcurrentQueue(10);
    q.put(10);
    q.put(10);
    q.put(10);
    q.put(1);
    q.put(1);
    q.put(1);
    q.put(1);
    q.put(0);
    q.put(0);
    q.put(0);
    q.put(0);
    q.put(0);
    print(q.size())
    print(q.get())
    print(q.size())

if __name__ == '__main__':
    if DEBUG:
        debug()
