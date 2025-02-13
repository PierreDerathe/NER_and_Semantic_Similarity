import torch
import torch.nn as nn
import torch.nn.functional as F
from gensim.models import Word2Vec
import numpy as np

class NERModelCNN(nn.Module):
    def __init__(self, pretrained_embeddings, num_classes, dropout=0.2):
        super(NERModelCNN, self).__init__()
        self.name = "cnn"
        self.num_classes = num_classes
        
        # Load pretrained embeddings
        embedding_size = pretrained_embeddings.shape[1]
        vocab_size = pretrained_embeddings.shape[0]
        
        self.embed = nn.Embedding.from_pretrained(
            torch.FloatTensor(pretrained_embeddings),
            padding_idx=0,
            freeze=False
        )
        
        # CNN parameters
        self.conv1 = nn.Conv1d(embedding_size, 128, 3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embed(x)  # (batch, seq_len, embed_dim)
        x = x.transpose(1, 2)  # (batch, embed_dim, seq_len)
        x = F.relu(self.conv1(x))  # (batch, hidden_size, seq_len)
        x = x.transpose(1, 2)  # (batch, seq_len, hidden_size)
        x = self.dropout(x)
        x = self.fc1(x)  # (batch, seq_len, num_classes)
        return x