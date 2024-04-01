import torch
import os
import platform

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
OS_TYPE = platform.system()

BATCH_SIZE = 256
NUM_WORKERS = 4
MODEL_DIR = {'base': './models/baseline.pt',
             'resnet': './models/resnet.pt'}

NUM_WORKERS = NUM_WORKERS if OS_TYPE == 'Linux' else 0
print("Device: {}\nOS Type: {}\n\nBatch size: {}\nWorkers: {}\n"
      .format(DEVICE, OS_TYPE, BATCH_SIZE, NUM_WORKERS))

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

transform_funcs = {
    'base': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ]),
    'resnet': transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomCrop(52),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

test_images = {x: ImageFolder(root='./splitted/test',
                              transform=transform_funcs[x]) for x in ['base', 'resnet']}

test_loader = {x: DataLoader(test_images[x],
                             batch_size=BATCH_SIZE,
                             shuffle=True,
                             num_workers=4) for x in ['base', 'resnet']}

import torch.functional as F

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


while True :
    mode = input("Evaluate what? [0: base | 1: resnet] >>")
    if mode in ['0', '1'] :
        mode = {'0': 'base', '1': 'resnet'}[mode]
        break

if os.path.exists(MODEL_DIR[mode]) :
    model = torch.load(MODEL_DIR[mode])
    model.eval()
    test_loss, test_accuracy = evaluate(model, test_loader[mode])
    print("{} Test Accuracy: {}%".format(mode[0].upper()+mode[1:], test_accuracy*100))
else :
    print("{} not learned. Learn the model first.".format(mode[0].upper()+mode[1:]))