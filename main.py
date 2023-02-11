from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import numpy as np
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim

from resnet import ResNet18

import utils

transform = utils.get_transform()
testloader = utils.get_testloader(transform)
# Load the CIFAR-10 dataset
trainset = utils.get_trainset(transform)
mean,std= utils.get_mean(trainset)
mean = np.mean(mean)
trainset = [
        (utils.apply_albumentations(data,mean,std)
    ,
    torch.tensor(target) ) 
for data, target in trainset]
device = utils.get_device()
model = ResNet18().to(device)
utils.get_summary(model)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2,pin_memory=True)


def train_and_test():
  #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
  optimizer =optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
  EPOCHS = 20
  scheduler = OneCycleLR(optimizer, max_lr=0.001, epochs=EPOCHS,
                      steps_per_epoch=len(trainloader))
 
  for epoch in range(EPOCHS):
      print("EPOCH:", epoch)
      utils.train(model, device, trainloader, optimizer,scheduler, epoch,EPOCHS)
      utils.test(model, device, testloader)


train_and_test()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

wrong_images=torch.from_numpy(np.array([img for img in utils.wrong_img[:25]]))

visualization= utils.grad_cam(device,model,wrong_images,3)

npimg =wrong_images.numpy().transpose(0,2,3,1)

utils.show_image(npimg,visualization,utils.pred_label,utils.target_label,classes)