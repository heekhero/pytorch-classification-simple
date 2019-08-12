import sys
import os

root_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, os.path.join(root_path, 'lib'))

import torch.utils.data

from model.network import NetWorks
from utils.config import cfg, get_classes
from dataset.dataset import Dataset

classes = get_classes(train=False)
num_class = len(classes)

net = NetWorks(num_class)

print('move model to CUDA devices...')
if cfg.TEST.CUDA:
    net = net.cuda()
print('Networks initialze successfully!')

params = 0
for param in net.parameters():
    params += torch.numel(param)
print('The number of parameters in this networks is {:.2f}M.'.format(float(params)/1e6))

net.eval()

print('start to load model parameters...')
model_dict = torch.load(cfg.TEST.LOAD_PATH)
net.load_state_dict({k:v for k,v in model_dict.items() if k in net.state_dict()})
print('done')

dataset = Dataset(classes, trainging=False)
num_data = len(dataset)
dataloader = torch.utils.data.DataLoader(dataset=dataset, shuffle=False, batch_size=1, num_workers=4)

dataiter = iter(dataloader)

print('start to evaluate the trained model')
sign = 0
for i in range(num_data):
    data = next(dataiter)
    input, label = data[0], data[1]
    if cfg.TEST.CUDA:
        input = input.cuda()
        label = label.cuda()

    output = net(input)
    predict = torch.argmax(output, dim=1).view(-1)
    if predict == label:
        sign += 1
    sys.stdout.write('eval process {}/{}'.format(i, num_data) + '\r')
    sys.stdout.flush()

precision = float(sign) / num_data

print('The precision of your model is {}'.format(precision))
