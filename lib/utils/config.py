import os
import pprint

from easydict import EasyDict as edict


__C = edict()

cfg = __C

__C.ROOT_DIR = os.path.abspath(os.path.join(__file__, '..', '..', '..'))


__C.TRAIN = edict()
__C.TRAIN.NUM_WORKS = 4  #the number of works when loading data used in Pytorch's DataLoader
__C.TRAIN.BATCH_SIZE = 32  #training batch size
__C.TRAIN.EPOCHS = 100  #total training epochs
__C.TRAIN.START_EPOCH = 1  #start of training epochs
__C.TRAIN.OPTIMIZER = 'adam'  #optimization method to train the networks, for now only adam method is supported
__C.TRAIN.LEARNING_RATE = 0.000001
__C.TRAIN.BETA1 = 0.9  # beta1 parameter of adam
__C.TRAIN.WEIGHT_DECAY = 0.0005
__C.TRAIN.USE_TFBOARD = True  #whether to use tensorboard for visualization
__C.TRAIN.DISP_FREQ = 50  #iters to print and plot training loss
__C.TRAIN.CUDA = True  #whether to use CUDA devices during training
__C.TRAIN.SAVE_EPOCH = 10
__C.TRAIN.LR_DECAY_EPOCH = 50
__C.TRAIN.LR_DECAY_GAMMA = 0.1
__C.TRAIN.RESIZE = 256
__C.TRAIN.CROP = 224
__C.TRAIN.INIT_METHOD = 'norm'
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.CONTINUE_TRAIN = True
__C.TRAIN.LOAD_PATH = os.path.join(__C.ROOT_DIR, 'output', 'net_10.pkl')

__C.TEST = edict()

__C.TEST.LOAD_PATH = os.path.join(__C.ROOT_DIR, 'output', 'net_40.pkl')  #path to load the trained model, please just replace xxx.pkl with your own model name
__C.TEST.CUDA = True  #whether to use CUDA devices during testing
__C.TEST.RESIZE = 256

__C.VGG16_PATH = '/path/to/vgg16_caffe.pth'

pprint.pprint(cfg)

def get_classes(train=True):
    if train:
        img_folder = os.path.join(cfg.ROOT_DIR, 'data', 'train')
    else:
        img_folder = os.path.join(cfg.ROOT_DIR, 'data', 'test')

    classes = []
    for root, dir, file in os.walk(img_folder):
        classes += dir

    print(classes)
    return sorted(classes)


