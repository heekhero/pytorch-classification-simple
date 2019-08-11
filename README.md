# pytorch-classification-simple
### :boom: This is a repository for image classification in Pytorch with very very very simple code! It is a very suitable entry project for beginners to learn in terms of deep learning.

In this repository, some important features of a deep classification model are written by the author manually instead of calling pytorch library directly e.g. the softmax cross entropy loss in training process and a number of connected [dense blocks](https://arxiv.org/abs/1608.06993) which are use to define the base architecture of netowrks.

### Features
* **It's pure Python code.** That means a beginner can easily understand the workflow of it and there will not be any confused and terrible C++ code for you to get confused!!!
* **Hyperparameters are flexible to change.** We took the method used by [rbgirshick/py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn). In the form of [easydict](https://pypi.org/project/easydict/), all the parameters needed were set in config file. You can set some parameters freely to debug before the program runs.
