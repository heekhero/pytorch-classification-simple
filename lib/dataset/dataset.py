import torch
import torchvision.transforms.functional
import torchvision.transforms as transforms
import os
import PIL.Image as Image
import numpy as np


from utils.config import cfg


class Dataset:
    def __init__(self, classes, trainging = True):
        self._classes = classes
        self.training = trainging
        self._num_class = len(self._classes)
        self._class_to_inds = dict(zip(self._classes, torch.arange(self._num_class)))
        # print(self._class_to_inds)
        self._data_dir = os.path.join(cfg.ROOT_DIR, 'data')
        self._img_dir = os.path.join(self._data_dir, 'train') if self.training else os.path.join(self._data_dir, 'test')
        self._train_data = self.get_train_data(self._classes)
        self._num_data = len(self._train_data)

        self._img_path = [self._train_data[i][0] for i in range(self._num_data)]
        self._ratio = self.get_img_ratio(self._img_path)
        self._ratio_inds = np.argsort(self._ratio)
        self._ratio_avg = self.get_img_ratio_avg(self._ratio_inds)

        self._data_mean, self._data_std = self.get_mean_and_std()
        # self._data_mean, self._data_std = (0.49, 0.46, 0.42), (0.26, 0.25, 0.25)
        self._transforms = transforms.Compose([transforms.Resize(cfg.TRAIN.RESIZE),
                                               transforms.RandomCrop(cfg.TRAIN.CROP),
                                               transforms.RandomHorizontalFlip(),
                                               # transforms.RandomV0erticalFlip(),
                                               transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                                               transforms.ToTensor(),
                                               transforms.Normalize(self._data_mean, self._data_std)])

    @property
    def ratio_inds(self):
        return self._ratio_inds

    def __getitem__(self, index):
        data_list = self._train_data[index]
        img = Image.open(data_list[0]).convert('RGB')
        img_array = np.array(img)
        img_array = img_array[:, :, ::-1]
        img = Image.fromarray(img_array)
        # if self._ratio_avg[index] < 1:
        #     self._transforms.transforms[0] = transforms.Resize((cfg.TRAIN.RESIZE, int(cfg.TRAIN.RESIZE / self._ratio_avg[index])))
        # else:
        #     self._transforms.transforms[0] = transforms.Resize((int(cfg.TRAIN.RESIZE * self._ratio_avg[index]), cfg.TRAIN.RESIZE))
        data = self._transforms(img)

        return data, data_list[1]

    def __len__(self):
        return self._num_data


    def get_img_ratio(self, img_paths):
        ratios = []
        for img_path in img_paths:
            img = Image.open(img_path).convert('RGB')
            width = img.size[0]
            height = img.size[1]
            ratio = float(width) / height
            ratios.append(ratio)
        return np.array(ratios)

    def get_train_data(self, classes):
        print('loading image paths')
        if not isinstance(classes, list):
            if isinstance(classes, dict) or isinstance(classes, tuple) or isinstance(classes, set):
                classes = list(classes)
        num_list= []
        for _class in classes:
            for root, _, files in os.walk(os.path.join(self._img_dir, _class)):
                for file in files:
                    if isimagefile(file):
                        data_list = []
                        path = os.path.join(root, file)
                        data_list.append(path)
                        data_list.append(self._class_to_inds[_class])
                        num_list.append(data_list)
        print('done')
        return num_list

    def get_mean_and_std(self):
        #get mean and std of training images
        print('calculate mean and std of training images...')
        images = [self._train_data[i][0] for i in range(self._num_data)]
        img_mean = 0
        for path in images:
            img = Image.open(path).convert('RGB')
            img_tensor = torchvision.transforms.functional.to_tensor(img)
            mean = torch.mean(torch.mean(img_tensor, dim=2), dim=1)
            img_mean += mean
        img_mean = img_mean / self._num_data

        img_std = 0
        for path in images:
            img = Image.open(path).convert('RGB')
            img_tensor = torchvision.transforms.functional.to_tensor(img)
            img_area = img_tensor.size(1) * img_tensor.size(2)
            data_img_mean = img_mean.reshape(-1, 1, 1).expand_as(img_tensor)
            std_this_img = torch.sqrt(torch.sum(torch.pow((data_img_mean - img_tensor), 2), dim=(1, 2)) / img_area)
            img_std += std_this_img
        img_std = img_std / self._num_data

        print('mean and std of training data are ({:.2f}/{:.2f}/{:.2f}), ({:.2f}/{:.2f}/{:.2f})'.format(img_mean[0], img_mean[1], img_mean[2],
                                                                                img_std[0], img_std[1], img_std[2]))
        return img_mean, img_std

    def get_img_ratio_avg(self, ratio_inds):
        bs = cfg.TRAIN.BATCH_SIZE
        ratios = self._ratio.copy()
        num_batches = int(np.ceil(len(ratio_inds) / bs))
        for i in range(num_batches):
            ratio_ins = ratio_inds[i*bs : min((i+1)*bs, len(ratio_inds))]
            ratios[ratio_ins] = np.mean(self._ratio[ratio_ins])
        return ratios

IMG_EXTENSION = ['jpg', 'png', 'jpeg']

def isimagefile(file):
    return any(file.endswith(extension) for extension in IMG_EXTENSION)