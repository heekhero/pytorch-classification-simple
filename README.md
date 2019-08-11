# pytorch-classification-simple
### :boom: This is a repository for image classification in Pytorch with very very very simple code! It is a very suitable entry project for beginners to learn in terms of deep learning.

In this repository, some important features of a deep classification model are written by the author manually instead of calling pytorch library directly e.g. the softmax cross entropy loss in training process and a number of connected [dense blocks](https://arxiv.org/abs/1608.06993) which are use to define the base architecture of netowrks.

## Good Features
* **It's pure Python code.** That means a beginner can easily understand the workflow of it and there will not be any confused and terrible C++ code for you to get confused!!!
* **Hyperparameters are flexible to change.** We took the method used by [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn). In the form of [easydict](https://pypi.org/project/easydict/), all the parameters needed were set in config file. You can set some parameters freely to debug before the program runs.
* **Tensorboard Support** To visualize the training process,  we introduce tensorboardX to our porject for a better visualization experience.

## Future Work
- [ ] Multiple GPUs training. Because the memory usage of general classification tasks are considerable due to the large minibatch, we will add some codes to support multiple GPUs training mode to accelerate the trainingn process and further enlarge the size of a minibatch as much as possible.
- [ ] Customizing image preprocessing process. In our now version of code, we simply use the[ Compose](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=compose#torchvision.transforms.Compose) function of Torchvision to generate the method of image preprocessing process. In the future vision, we will replace them with our lambda process function

## Installation
Firstly, clone the entire project
```
git  clone  https://github.com/heekhero/pytorch-classification-simple/blob/master/README.md
```

### Prerequisites
* Python 3.6
* Pytorch 1.1.0 and torchvision 0.3.0 [Installation instructions](https://pytorch.org/get-started/locally/)
* CUDA 9.0 or higher

### Data Preparation
create data folders:
```
cd classification
mkdir data
cd data
mkdir train
mkdir test
```

Prepare your own dataset in form of folder + images and I highly recommended you to create soft link with respect to these prepared datasets in train and test folder you have created before.

The tree architecture of the folders should look like this:
```
data
    train
        class_1
            xxx.jpg
            yyy.jpg
            zzz.jpg
            ...
        class_2
            xxx.jpg
            yyy.jpg
            zzz.jpg
            ...
        ...
    test
        class_1
            xxx.jpg
            yyy.jpg
            zzz.jpg
            ...
        class_2
            xxx.jpg
            yyy.jpg
            zzz.jpg
            ...
        ...
```
PS: class_X is the true class of your images e.g. person, train, bike... and the file name of images is not important but need to follow the correct image suffix. For simplification, run the follow instrction to prepare a simple datasets in terms of Apple and Orange classification.
```

```
