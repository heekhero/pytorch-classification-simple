import torchvision.models as models
import torch.nn as nn
import torch
import torch.nn.init as init

from utils.config import cfg

class vgg16(nn.Module):
    def __init__(self, pre_trained, num_classes):
        super(vgg16, self).__init__()
        self._vgg_path = cfg.VGG16_PATH
        self._pre_trained = pre_trained
        self._num_classes = num_classes

        self._init_modules()
        self._init_parameters()

    def forward(self, input):
        out = self._feature(input)
        out = self._pooling(out)
        out = out.reshape(-1, (int(cfg.TRAIN.CROP / 32) ** 2) * 512)
        out = self._classifier(out)
        out = torch.sigmoid(out)

        return out

    def _init_modules(self):
        _vgg16 = models.vgg16_bn()

        if self._pre_trained:
            print('Load pretrained parameters from {}'.format(self._vgg_path))
            state_dict = torch.load(self._vgg_path)
            _vgg16.load_state_dict({k:v for k,v in state_dict.items() if k in _vgg16.state_dict()})
            print('done')

        self._feature = _vgg16.features

        for layer in range(14):
            for p in self._feature[layer].parameters(): p.requires_grad = False

        self._pooling = _vgg16.avgpool
        self._classifier = nn.Sequential(*(list(_vgg16.classifier._modules.values())[:-1] +
                                         [nn.Linear(in_features=4096, out_features=self._num_classes)]))

    def _init_parameters(self):
        def module_init(m):
            classname =  m.__class__.__name__
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                if hasattr(m, 'weight'):
                    if cfg.TRAIN.INIT_METHOD == 'xavier':
                        init.xavier_normal_(m.weight, 1)
                    elif cfg.TRAIN.INIT_METHOD == 'norm':
                        init.normal_(m.weight, 0, 1)
                    elif cfg.TRAIN.INIT_METHOD == 'kaiming':
                        init.kaiming_normal_(m.weight, nonlinearity='relu')
                    else:
                        raise NotImplementedError
                if hasattr(m, 'bias'):
                    init.constant_(m.bias, 0)
            elif classname.find('Norm') != -1:
                if hasattr(m, 'weight'):
                    init.normal_(m.weight, 0, 1)
                if hasattr(m, 'bias'):
                    init.constant_(m.bias, 0)

        print('initialize networks with {} ...'.format(cfg.TRAIN.INIT_METHOD))
        self._feature.apply(module_init)
        self._pooling.apply(module_init)
        self._classifier.apply(module_init)
        print('done')
