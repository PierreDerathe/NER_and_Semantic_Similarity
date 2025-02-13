import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_conll(file_path):
    """Convert CONLL file to format suitable for CNN model"""
    sentences = []
    labels = []
    current_sentence = []
    current_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split()
                if len(parts) >= 5:
                    current_sentence.append(parts[1])
                    current_labels.append(parts[4])
            elif current_sentence:
                sentences.append(' '.join(current_sentence))
                labels.append(current_labels)
                current_sentence = []
                current_labels = []
    
    # Handle last sentence if file doesn't end with empty line
    if current_sentence:
        sentences.append(' '.join(current_sentence))
        labels.append(current_labels)
    
    # Create DataFrame
    df = pd.DataFrame({
        'review': sentences,
        'label': labels
    })
    
    # Create and fit label encoder on all labels
    label_encoder = LabelEncoder()
    # Flatten all labels to fit encoder
    all_labels = [label for label_seq in labels for label in label_seq]
    label_encoder.fit(all_labels)
    
    # Transform each sequence
    df['label'] = df['label'].apply(lambda x: label_encoder.transform(x).tolist())
    
    return df, label_encoder