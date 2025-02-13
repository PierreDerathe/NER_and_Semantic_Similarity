import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
import numpy as np

class NERModelCNN(nn.Module):
    def __init__(self, pretrained_embeddings, num_classes, dropout=0.2):
        super(NERModelCNN, self).__init__()
        self.name = "cnn"
        
        # Load pretrained embeddings
        embedding_size = pretrained_embeddings.shape[1]
        vocab_size = pretrained_embeddings.shape[0]
        
        self.embed = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_embeddings),
            padding_idx=0,
            freeze=False  # Allow fine-tuning
        )
        
        # CNN parameters
        Ci = 1
        Co = 100
        Ks = [3,4,5]
        
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Co, (K, embedding_size)) for K in Ks])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, num_classes)

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)
        x = x.unsqueeze(1)  # (N, Ci, W, D)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit