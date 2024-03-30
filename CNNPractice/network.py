import torch

class CNN(torch.nn.Module) :

    def __init__(self) :
        super(CNN, self).__init__()
        self.keep_prob = 0.5

        '''
        [Layer 1] CONV
           INPUT => (batch, 28, 28, 1)
        => CONV  => (batch, 26, 26, 32)
        => PADD  => (batch, 28, 28, 32)
        => ReLU
        => POOL  => (batch, 14, 14, 32)
        => OUTPUT
        '''
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        '''
        [Layer 2] CONV
           INPUT => (batch, 14, 14, 32)
        => CONV  => (batch, 12, 12, 64)
        => PADD  => (batch, 14, 14, 64)
        => ReLU
        => POOL  => (batch, 7, 7, 64)
        => OUTPUT
        '''
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        '''
        [Layer 3] CONV
           INPUT => (batch, 7, 7, 64)
        => CONV  => (batch, 5, 5, 128)
        => PADD  => (batch, 7, 7, 128)
        => ReLU
        => POOL  => (batch, 3, 3, 128)
        => PADD  => (batch, 4, 4, 128)
        => OUTPUT
        '''
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        '''
        [Layer 4] FCON
           INPUT  => (batch, 4*4*128)
        => LINEAR => (batch, 625)
        => ReLU
        => DROPOUT [RATE]
        => OUTPUT
        '''

        self.fc1 = torch.nn.Linear(4*4*128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)

        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p=1-self.keep_prob)
        )

        '''
        [Layer 5] FCON
        => LINEAR => (batch, 10)
        => ReLU
        => DROPOUT [RATE]
        => OUTPUT
        '''

        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
    
    
    def forward(self, x) :
        '''
        [OVERALL]
           INPUT  => (batch, 28, 28, 1)
        => [L1]   => (batch, 14, 14, 32)
        => [L2]   => (batch, 7, 7, 64)
        => [L3]   => (batch, 4, 4, 128)
        => FLAT   => (batch, 4*4*128)
        => [L4]   => (batch, 625)
        => [L5]   => (batch, 10)
        => OUTPUT
        '''

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1) #Flatten1D
        out = self.layer4(out)
        out = self.fc2(out)
        return out
