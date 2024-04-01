import torch
import os
import platform

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
OS_TYPE = platform.system()

BATCH_SIZE = 256
EPOCH = 30
NUM_WORKERS = 4
MODEL_DIR = './models/resnet.pt'

NUM_WORKERS = NUM_WORKERS if OS_TYPE == 'Linux' else 0
print("Device: {}\nOS Type: {}\n\nBatch size: {}\nEpoch: {}\nWorkers: {}\n"
      .format(DEVICE, OS_TYPE, BATCH_SIZE, EPOCH, NUM_WORKERS))

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(52),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomCrop(52),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

data_dir = './splitted'
image_datasets = {x: ImageFolder(root=os.path.join(data_dir, x),
                                 transform=data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: DataLoader(image_datasets[x],
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=NUM_WORKERS) for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

class_names = image_datasets['train'].classes

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models

resnet = models.resnet50(pretrained=True)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 33)

if os.path.exists(MODEL_DIR) :
    resnet.load_state_dict(torch.load(MODEL_DIR, map_location=DEVICE))
    resnet.train()
resnet = resnet.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad,
                                 resnet.parameters()), lr=0.001)

from torch.optim import lr_scheduler

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft,
                                       step_size=7,
                                       gamma=0.1)

ct = 0
for child in resnet.children() :
    ct += 1
    if ct < 6 :
        for param in child.parameters() :
            param.requires_grad = False

import time
import copy

def train_resnet(model, criterion, optimizer, scheduler, num_epochs = 25) :

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs) :
        print('---------------<Epoch {}>---------------'.format(epoch+1))
        since = time.time()

        for phase in ['train', 'val'] :
            if phase == 'train' :
                model.train()
            else :
                model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase] :
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train') :
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train' :
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' :
                scheduler.step()
                l_r = [x['lr'] for x in optimizer_ft.param_groups]
                print("Learning rate: {}".format(l_r))
            
            epoch_loss = running_loss/dataset_sizes[phase]
            epoch_acc = running_corrects.double()/dataset_sizes[phase]
        
            print("{} Loss: {:.4f} | Accuracy: {:.4f}%".format(phase, epoch_loss, epoch_acc*100))

            if phase == 'val' and best_acc < epoch_acc :
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))
    
    print('Best Validation Accuracy: {}%'.format(best_acc*100))

    model.load_state_dict(best_model_wts)
    return model

model_resnet50 = train_resnet(resnet, criterion, optimizer_ft, \
                              exp_lr_scheduler, num_epochs=EPOCH)

if OS_TYPE == 'Linux' :
    torch.save(model_resnet50, MODEL_DIR)
elif OS_TYPE == 'Windows' :
    torch.save(model_resnet50.state_dict(), MODEL_DIR)

print("model saved!")