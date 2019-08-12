import os
import pprint

from easydict import EasyDict as edict


__C = edict()

cfg = __C

__C.ROOT_DIR = os.path.abspath(os.path.join(__file__, '..', '..', '..'))


__C.TRAIN = edict()
__C.TRAIN.NUM_WORKS = 4  #the number of works when loading data used in Pytorch's DataLoader
__C.TRAIN.BATCH_SIZE = 32  #training batch size
__C.TRAIN.EPOCHS = 200  #total training epochs
__C.TRAIN.START_EPOCH = 1  #start of training epochs
__C.TRAIN.OPTIMIZER = 'adam'  #optimization method to train the networks, for now only adam method is supported
__C.TRAIN.LEARNING_RATE = 0.003
__C.TRAIN.BETA1 = 0.5  # beta1 parameter of adam
__C.TRAIN.WEIGHT_DECAY = 0.0005
__C.TRAIN.USE_TFBOARD = True  #whether to use tensorboard for visualization
__C.TRAIN.DISP_FREQ = 100  #iters to print and plot training loss
__C.TRAIN.CUDA = True  #whether to use CUDA devices during training

__C.TEST = edict()

__C.TEST.LOAD_PATH = os.path.join(__C.ROOT_DIR, 'output', 'xxx.pkl')  #path to load the trained model, please just replace xxx.pkl with your own model name
__C.TEST.CUDA = True  #whether to use CUDA devices during testing

pprint.pprint(cfg)

def get_classes(train=True):
    if train:
        img_folder = os.path.join(cfg.ROOT_DIR, 'data', 'train')
    else:
        img_folder = os.path.join(cfg.ROOT_DIR, 'data', 'test')

    classes = []
    for root, dir, file in os.walk(img_folder):
        classes += dir

    return classes


