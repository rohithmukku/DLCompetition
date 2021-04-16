# -*- coding: utf-8 -*-
"""simclr_resnet50.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zw7Oxuhlg-uxaQbTZuDwvzF-DBlv2rMz
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import torch
import torch.optim as optim
from torch.autograd import Variable
from PIL import Image # PIL is a library to process images
import os
import torchvision
from tqdm.notebook import tqdm

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /dataset
            split: The split you want to used, it should be one of train, val or unlabeled.
            transform: the transform you want to applied to the images.
        """

        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(root, split)
        label_path = os.path.join(root, f"{split}_label_tensor.pt")

        self.num_images = len(os.listdir(self.image_dir))

        if os.path.exists(label_path):
            self.labels = torch.load(label_path)
        else:
            self.labels = -1 * torch.ones(self.num_images, dtype=torch.long)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        with open(os.path.join(self.image_dir, f"{idx}.png"), 'rb') as f:
            img = Image.open(f).convert('RGB')

        return self.transform(img), self.labels[idx]

# Commented out IPython magic to ensure Python compatibility.
train_dataset = CustomDataset(root="dataset", split="train", transform=transforms.ToTensor())
test_dataset = CustomDataset(root="dataset", split="val", transform=transforms.ToTensor())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

model = torchvision.models.resnet50(pretrained=False, num_classes=800).to(device)

checkpoint = torch.load('/content/drive/MyDrive/checkpoint_0150.pth.tar', map_location=device)
state_dict = checkpoint['state_dict']

i = 0
for k in list(state_dict.keys()):
  print(k)
  i += 1
  if k.startswith('module.backbone.'):
    if k.startswith('module.backbone') and not k.startswith('module.backbone.fc'):
      # remove prefix
      state_dict[k[len("module.backbone."):]] = state_dict[k]

  del state_dict[k]

for k in list(state_dict.keys()):
  print(k)

# print(checkpoint)
log = model.load_state_dict(state_dict, strict=False)
print(log)
assert log.missing_keys == ['fc.weight', 'fc.bias']

labeled_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
unlabeled_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

# freeze all layers but the last fc
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False

parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
assert len(parameters) == 2  # fc.weight, fc.bias

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0008)
criterion = torch.nn.CrossEntropyLoss().to(device)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

epochs = 25
epochs = tqdm(range(epochs))
print(device)
print(len(train_dataset), len(test_dataset))
for epoch in epochs:
  top1_train_accuracy = 0
  for counter, (x_batch, y_batch) in enumerate(labeled_loader):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    logits = model(x_batch)
    loss = criterion(logits, y_batch)
    
    top1, top5 = accuracy(logits, y_batch, topk=(1,5))
    top1_train_accuracy += top1[0]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
  top1_train_accuracy /= (counter + 1)
  top1_accuracy = 0
  top5_accuracy = 0
  for counter, (x_batch, y_batch) in enumerate(unlabeled_loader):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    logits = model(x_batch)
  
    top1, accuracy(logits, y_batch, topk=(1,))
    top1_accuracy += top1[0]
    top5_accuracy += top5[0]
  
  top1_accuracy /= (counter + 1)
  top5_accuracy /= (counter + 1)
  print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")
torch.save(model.state_dict(),"$SCRATCH/finalcheckpoint_50_100.pth.tar")

# model1 = torchvision.models.resnet18(pretrained=False, num_classes=800).to(device)
# model1.load_state_dict(torch.load('/content/drive/MyDrive/model.pth', map_location=device))
# model1.eval()
# top1_accuracy = 0
# top5_accuracy = 0
# for counter, (x_batch, y_batch) in enumerate(unlabeled_loader):
#   x_batch = x_batch.to(device)
#   y_batch = y_batch.to(device)

#   logits = model1(x_batch)
  
#   top1, top5 = accuracy(logits, y_batch, topk=(1,5))
#   top1_accuracy += top1[0]
#   top5_accuracy += top5[0]
#   print(counter)
  
# top1_accuracy /= (counter + 1)
# top5_accuracy /= (counter + 1)
# print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()}\tTop1 Test accuracy: {top1_accuracy.item()}\tTop5 test acc: {top5_accuracy.item()}")