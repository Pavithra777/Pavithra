from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import numpy as np
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from tqdm import tqdm
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import matplotlib.pyplot as plt
from torchsummary import summary

wrong_img =[]
pred_label =[]
target_label=[]
wrong_preds=[]

def get_device():
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  return device

def apply_albumentations(img,mean,std):
    img =img.permute(1, 2, 0)
    img=np.array(img)
    aug = A.Compose([
	#A.Cutout(num_holes=1, max_h_size=5, max_w_size=5, always_apply=True, p=0.2),
	 A.CoarseDropout(max_holes=1,
                       min_holes =1,
                       max_height=5, 
                       max_width=5, 
                       p=0.2,
                       fill_value= mean,
                       min_height=5, min_width=5, always_apply=False)
    ,A.PadIfNeeded(min_height=36, min_width=36, p=0.2)
    ,A.RandomCrop(32, 32, always_apply=True, p=0.2)
  #,A.Normalize(mean=mean, std=std, always_apply=True)
    ,ToTensorV2()

   ])
    augmented_img = aug(image=img)['image']
    return augmented_img

def get_transform():
  return transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_transform1():
  return transforms.Compose([
     transforms.ToTensor()
])

def get_trainset(transform):
  # Load the CIFAR-10 dataset
  return datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform )

def get_testloader(transform):
  testset= torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
  return torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2,pin_memory=True)

def get_summary(model):
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  summary(model, input_size=(3, 32, 32))


def train(model, device, trainloader, optimizer,scheduler, epoch,lastepoch,loss_l1 =False):
  model.train()
  
  pbar = tqdm(trainloader)
  
  correct = 0
  processed = 0
  for batch_idx, (data, target) in enumerate(pbar):
    
    # get samples
    data, target = data.to(device), target.to(device)
    # Init
    optimizer.zero_grad()
    
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    y_pred = model(data)
    
    # Calculate loss

    loss = F.nll_loss(y_pred, target)
    if(loss_l1):
      l1=0
      lamba_l1 = 1e-5
      for p in model.parameters():
        l1=l1+p.abs().sum()
      loss = loss+lamba_l1*l1
  
    # Backpropagation
    loss.backward()
    optimizer.step()
    scheduler.step()
    # Update pbar-tqdm
    
    pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(data)
    pbar.set_description(desc= f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')
    if epoch == lastepoch-1 :
      wrong_preds = [i for i, x in enumerate(pred.eq(target.view_as(pred))) if x == False]
      tar = target.reshape(-1, 1)
      for index_val in wrong_preds:
        if len(wrong_img) < 10:
          wrong_img.append(data[index_val].cpu().numpy().squeeze())
          target_label.append(tar[index_val].cpu().numpy().squeeze())
          pred_label.append(pred[index_val].cpu().numpy().squeeze())

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
   
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def grad_cam(device,model,images,class_num):

  target_layers = [model.layer4[-1]]
  input_tensor = images
  # Note: input_tensor can be a batch tensor with several images!

  # Construct the CAM object once, and then re-use it on many images:
  cam = GradCAM(model=model, target_layers=target_layers, use_cuda=device)

  # You can also use it within a with statement, to make sure it is freed,
  # In case you need to re-create it inside an outer loop:
  # with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
  #   ...

  # We have to specify the target we want to generate
  # the Class Activation Maps for.
  # If targets is None, the highest scoring category
  # will be used for every image in the batch.
  # Here we use ClassifierOutputTarget, but you can define your own custom targets
  # That are, for example, combinations of categories, or specific outputs in a non standard model.

  targets = [ClassifierOutputTarget(class_num)]

  # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
  grayscale_cam = cam(input_tensor=input_tensor, targets=targets,aug_smooth=True,eigen_smooth=True)

  # In this example grayscale_cam has only one image in the batch:
  grayscale_cam = grayscale_cam[0, :]
  npimg=input_tensor.numpy().transpose(0,2,3,1)
  visualization = show_cam_on_image(npimg, grayscale_cam)
  return visualization

def show_image(original_img,grad_cam_visulaisation,pred_lable,target_label,classes):
  X2 = grad_cam_visulaisation
  X1 = original_img
  Y = pred_label
  #Visualizing CIFAR 10
  fig, axes1 = plt.subplots(4,5,figsize=(32,32))
  rnlist = []
  for j in range(0,4,2):
      for k in range(5):
          i=-1
          while (True):
            i=np.random.choice(range(len(X1)))        
            if i not in rnlist:
              rnlist.append(i)
              break 
          axes1[j][k].set_axis_off()
          axes1[j][k].imshow(X1[i:i+1][0])
          axes1[j][k].set_title("True: %s\nPredict: %s" % (classes[target_label[i]], classes[Y[i]]))
          axes1[j+1][k].imshow(X2[i:i+1][0])
          axes1[j+1][k].set_title("True: %s\nPredict: %s" % (classes[target_label[i]], classes[Y[i]]))
  


def get_mean(trainset):
  return trainset.data.mean(axis=(0, 1, 2)) / 255, trainset.data.std(axis=(0, 1, 2)) / 255
   