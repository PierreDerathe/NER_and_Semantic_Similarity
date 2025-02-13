import torch
from gensim.models import Word2Vec
from preprocess_conll import preprocess_conll
from ner_cnn_pretrained import NERModelCNN
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import argparse
import os

def load_pretrained_embeddings(word2vec_model, word2idx, embedding_dim=100):
    embedding_matrix = np.zeros((len(word2idx), embedding_dim))
    for word, idx in word2idx.items():
        if word in word2vec_model.wv:
            embedding_matrix[idx] = word2vec_model.wv[word]
    return embedding_matrix

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default='QUAERO_FrenchMed/EMEA/EMEAtrain_layer1_ID.conll')
    parser.add_argument("--valid", default='QUAERO_FrenchMed/EMEA/EMEAdev_layer1_ID.conll')
    parser.add_argument("--test", default='QUAERO_FrenchMed/EMEA/EMEAtest_layer1_ID.conll')
    parser.add_argument("--embeddings", default='model/word2vec_cbow_med_train20.model')
    parser.add_argument("--epochs", default=5, type=int)
    args = parser.parse_args()

    # Load pre-trained Word2Vec model
    word2vec_model = Word2Vec.load(args.embeddings)

    # Preprocess CONLL data
    train_df, label_encoder = preprocess_conll(args.train)
    valid_df, _ = preprocess_conll(args.valid)
    test_df, _ = preprocess_conll(args.test)

    # Create vocabulary from Word2Vec model
    word2idx = {word: idx for idx, word in enumerate(word2vec_model.wv.index_to_key)}
    word2idx['<PAD>'] = len(word2idx)
    word2idx['<UNK>'] = len(word2idx)

    # Load embeddings
    embedding_matrix = load_pretrained_embeddings(word2vec_model, word2idx)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NERModelCNN(
        pretrained_embeddings=embedding_matrix,
        num_classes=len(label_encoder.classes_)
    ).to(device)

    # Training parameters
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch_size = 32

    # Results file
    results_file = "results.txt"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("NER Classification Results with Pre-trained Word2Vec Embeddings\n")
        f.write("=" * 50 + "\n\n")
        
        # Training loop
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            
            # Training metrics
            all_preds = []
            all_labels = []
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output.view(-1, num_classes), target.view(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                preds = output.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
            
            # Calculate metrics
            train_report = classification_report(all_labels, all_preds, target_names=label_encoder.classes_)
            
            f.write(f"Epoch {epoch+1}/{args.epochs}\n")
            f.write(f"Training Loss: {train_loss/len(train_loader):.4f}\n")
            f.write("Classification Report:\n")
            f.write(train_report)
            f.write("\n" + "="*30 + "\n")

if __name__ == "__main__":
    main()