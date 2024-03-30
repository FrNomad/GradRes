import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(torch.nn.Module) :

    def __init__(self, vocab_size, num_labels) :
        super(CNN, self).__init__()
        self.num_filter_sizes = 1
        self.num_filters = 256

        self.word_embed = torch.nn.Embedding(num_embeddings=vocab_size,
                                             embedding_dim=128,
                                             padding_idx=0)
        self.conv1 = torch.nn.Conv1d(128, self.num_filters, 5, stride=1)
        self.dropout = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(1*self.num_filters, num_labels, bias=True)

    def forward(self, inputs) :
        '''
        [OVERALL]
           INPUT   => (batch, 500, 1)
        => EMBED   => (batch, 500, 128)
        => PERM    => (batch, 128, 500)
        => CONV1D  => (batch, 256, 496)
        => PERM    => (batch, 496, 256)
        => POOL    => (batch, 256)
        => ReLU    => (batch, 256)
        => DROPOUT => (batch, 256)
        => LINEAR  => (batch, 2)
        '''

        embedded = self.word_embed(inputs).permute(0, 2, 1)
        x = F.relu(self.conv1(embedded).permute(0, 2, 1).max(1)[0])
        y_pred = self.fc1(self.dropout(x))

        return y_pred