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
                # Split line into columns (token_idx, token, _, _, ner_tag)
                parts = line.strip().split()
                if len(parts) >= 5:
                    current_sentence.append(parts[1])  # token
                    current_labels.append(parts[4])    # NER tag
            elif current_sentence:
                sentences.append(' '.join(current_sentence))
                labels.append(current_labels)
                current_sentence = []
                current_labels = []
    
    # Create DataFrame
    df = pd.DataFrame({
        'review': sentences,
        'label': labels
    })
    
    # Convert NER tags to numeric labels
    label_encoder = LabelEncoder()
    df['label'] = df['label'].apply(lambda x: label_encoder.fit_transform(x))
    
    return df, label_encoder