import os
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
from network import CNN

import matplotlib.pyplot as plt
import warnings


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

#################################################################

model = CNN()
model_path = 'model/cnn_model_b100.pth'

if os.path.exists(model_path) :
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
model.to(device)

#################################################################

warnings.filterwarnings(action='ignore')
with torch.no_grad() :
    x, y = 8, 4
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    y_test = mnist_test.test_labels.to(device)
    prediction = model(X_test)
    prediction = torch.argmax(prediction, 1)
    rand_keys = (torch.rand(x*y) * len(y_test)).int()

    print("Randomly chosen %d keys: %s" % (x*y, str(rand_keys)))

    max_grid_size = 12
    if x > y :
        res = (max_grid_size, max_grid_size*y/x)
    else :
        res = (max_grid_size*x/y, max_grid_size)
    fig, ax = plt.subplots(y, x, sharex=True, figsize=res)
    for y1 in range(y) :
        for x1 in range(x) :
            i = y1*x+x1
            ax[y1][x1].imshow(X_test[rand_keys[i]].view(28, 28), vmin = 0,\
                vmax = 255, cmap = 'Greys_r')
            ax[y1][x1].axis('off')
            ax[y1][x1].set_title(str(prediction[rand_keys[i]].item()))
    fig.show()
    plt.pause(60)