import os
import sys
import time
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from torch.utils.data import DataLoader
from network import CNN
from time_util import *

import matplotlib.pyplot as plt
import numpy as np
import warnings


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

torch.manual_seed(777)

if device == 'cuda' :
    torch.cuda.manual_seed_all(777)

model_path = 'model/cnn_model_smallbatch.pth'
learning_rate = 1e-3
training_epochs = int(input("Enter train epoch >> "))
batch_size = 64
print(f"Epochs: {training_epochs}")


mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                          train=False,
                          transform=transforms.ToTensor(),
                          download=True)

data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size,
                         shuffle=True,
                         drop_last=True)

#################################################################

model = CNN()
if os.path.exists(model_path) :
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.train()

model.share_memory()
model.to(device)


criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


total_batch = len(data_loader)
print(f"Total Batch Count: {total_batch}")

plt_X, plt_y = [ep+1 for ep in range(training_epochs)], [0] * training_epochs

#################################################################

print("[LEARN]")

def progressBar(size, progress) :
    bar_num = int(size*progress)
    return "[%s%s]"%('-'*bar_num,' '*(size-bar_num))

start_time = time.time()
loss_total = 0
for epoch in range(1, training_epochs + 1) :

    total_cycle, cycle = len(data_loader), 0
    epoch_loss = 0
    for X, y in data_loader :
        cycle += 1

        X = X.to(device)
        y = y.to(device)
        
        optimizer.zero_grad()
        hypothesis = model(X)
        loss = criterion(hypothesis, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        
        if cycle > 1 :
            sys.stdout.write('\r')
        total_progress = epoch/training_epochs
        epoch_progress = cycle/total_cycle
        time_elapsed = time.time() - start_time
        sys.stdout.flush()
        sys.stdout.write("Epoch %4d (%3d%%) : %s [%d/%d] (%3d%%) %s" % (epoch, total_progress * 100, progressBar(25, epoch_progress), 
                                                                    cycle, total_cycle, epoch_progress * 100, asMinutes(time_elapsed)))                             
        if cycle < total_cycle :
            try :
                sys.stdout.write(" (total exp : %s)" % asMinutes(time_elapsed*training_epochs/(epoch-1+(cycle-1)/total_cycle)))
            except :
                sys.stdout.write(" (total exp : %s)" % "inf")
            sys.stdout.write(' ' * 10)

    print(' -- loss: %.4f        ' % (epoch_loss/total_cycle))
    plt_y[epoch - 1] = epoch_loss/total_cycle


if training_epochs > 0 :
    print("Learn Completed! (%s)\n" % (asMinutes(time_elapsed)))

#################################################################

print("[TEST]")
model.eval()
with torch.no_grad() :
    warnings.filterwarnings(action='ignore')
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = (torch.argmax(prediction, 1) == y_test)
    accuracy = correct_prediction.float().mean()
    print("Accuracy: %.4lf %s" % (accuracy.item() * 100, '%'))

    plt.plot(plt_X, plt_y)
    plt.show()
    torch.save(model.state_dict(), model_path)
    print("Model saved!")