import _init_paths

from datasets.pascal_voc import pascal_voc
from utils.process_config import process_config
from utils.data_batch import databatch
def main():

    data_conf, common_conf = process_config("../config/train.cfg")

    conf={"phase":"train", "flipped": True}
    imdb = pascal_voc("PASCAL_VOC", "trainval", data_conf, common_conf, conf)
    imdb.prepare()
    data_batch = databatch(30, 10, imdb)
    data_batch.ready()
    imgs, lables = data_batch.next_batch()



if __name__ == "__main__":
    main()
