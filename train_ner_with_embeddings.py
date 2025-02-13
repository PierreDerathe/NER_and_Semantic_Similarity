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

def encode_sentences(sentences, word2idx, max_len=128):
    """Convert sentences to word IDs with padding"""
    encoded = []
    for sentence in sentences:
        # Split sentence into words and convert to IDs
        words = sentence.split()
        ids = [word2idx.get(word, word2idx['<UNK>']) for word in words]
        # Pad or truncate to max_len
        if len(ids) < max_len:
            ids = ids + [word2idx['<PAD>']] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
        encoded.append(ids)
    return encoded

def pad_sequences(sequences, pad_value, max_len=128):
    """Pad sequences to max_len"""
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded_seq = seq + [pad_value] * (max_len - len(seq))
        else:
            padded_seq = seq[:max_len]
        padded.append(padded_seq)
    return padded

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

    # Define max sequence length
    max_len = 128
    pad_token_id = word2idx['<PAD>']
    pad_label_id = -100  # Special padding index for labels

    # Convert and pad sequences
    train_word_ids = encode_sentences(train_df['review'].values, word2idx, max_len)
    valid_word_ids = encode_sentences(valid_df['review'].values, word2idx, max_len)
    test_word_ids = encode_sentences(test_df['review'].values, word2idx, max_len)

    # Pad labels
    train_labels_padded = pad_sequences(train_df['label'].values, pad_label_id, max_len)
    valid_labels_padded = pad_sequences(valid_df['label'].values, pad_label_id, max_len)
    test_labels_padded = pad_sequences(test_df['label'].values, pad_label_id, max_len)

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
    
    # Create data loaders
    train_data = torch.tensor(train_word_ids, dtype=torch.long)
    train_labels = torch.tensor(train_labels_padded, dtype=torch.long)
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_data = torch.tensor(valid_word_ids, dtype=torch.long)
    valid_labels = torch.tensor(valid_labels_padded, dtype=torch.long)
    valid_dataset = TensorDataset(valid_data, valid_labels)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

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
                # Reshape for loss calculation
                output = output.view(-1, model.num_classes)
                target = target.view(-1)
                # Create mask for non-padding tokens
                mask = target != pad_label_id
                output = output[mask]
                target = target[mask]
                # Calculate loss and update
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
                # Get predictions
                preds = output.argmax(dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
            
            # Calculate metrics
            train_report = classification_report(
                all_labels, 
                all_preds, 
                target_names=label_encoder.classes_,
                zero_division=0
            )
            
            f.write(f"Epoch {epoch+1}/{args.epochs}\n")
            f.write(f"Training Loss: {train_loss/len(train_loader):.4f}\n")
            f.write("Classification Report:\n")
            f.write(train_report)
            f.write("\n" + "="*30 + "\n")

if __name__ == "__main__":
    main()