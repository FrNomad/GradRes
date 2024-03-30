import torch
import os
import platform

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
OS_TYPE = platform.system()

BATCH_SIZE = 256
EPOCH = 30
NUM_WORKERS = 4
MODEL_DIR = './models/baseline.pt'

NUM_WORKERS = NUM_WORKERS if OS_TYPE == 'Linux' else 0
print("Device: {}\nOS Type: {}\n\nBatch size: {}\nEpoch: {}\nWorkers: {}\n"
      .format(DEVICE, OS_TYPE, BATCH_SIZE, EPOCH, NUM_WORKERS))

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

transform_base = transforms.Compose([transforms.Resize((64, 64)),
                                     transforms.ToTensor()])

train_dataset = ImageFolder(root='./splitted/train',
                            transform=transform_base)
val_dataset = ImageFolder(root='./splitted/val',
                            transform=transform_base)

from torch.utils.data import DataLoader


train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)

val_loader = DataLoader(val_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)

from neural_networks import Net
import torch.optim as optim
import torch.nn.functional as F

model_base = Net().to(DEVICE)
optimizer = optim.Adam(model_base.parameters(), lr=0.001)
if os.path.exists(MODEL_DIR) :
    model_base.load_state_dict(torch.load(MODEL_DIR, map_location=DEVICE))
    model_base.train()

def train(model, train_loader, optimizer) :
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader) :
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

def evaluate(model, test_loader) :
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad() :
        for data, target in test_loader :
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)

            test_loss += F.cross_entropy(output,
                                         target,
                                         reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

import time
import copy

def train_baseline(model, train_loader, val_loader, optimizer, num_epochs = 30) :
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1) :
        since = time.time()
        train(model, train_loader, optimizer)
        train_loss, train_acc = evaluate(model, train_loader)
        val_loss, val_acc = evaluate(model, val_loader)
        if val_acc > best_acc :
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        time_elapsed = time.time() - since
        print('---------------<Epoch {}>---------------'.format(epoch))
        print('Train Loss: {:.4f}, Accuracy: {:.2f}%'.format(train_loss, train_acc))
        print('Val   Loss: {:.4f}, Accuracy: {:.2f}%'.format(val_loss, val_acc))
        print('Completed in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))

    model.load_state_dict(best_model_wts)
    return model

base = train_baseline(model_base, train_loader,
                      val_loader, optimizer, EPOCH)

torch.save(base, MODEL_DIR)