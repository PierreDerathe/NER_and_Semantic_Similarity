# NER Classification with Pre-trained Word Embeddings

This project implements Named Entity Recognition (NER) using CNN models with pre-trained Word2Vec embeddings on medical text data.

## Project Structure

```
chatbot/
├── model/
│   └── word2vec_cbow_med_train20.model
├── QUAERO_FrenchMed/
│   └── EMEA/
│       ├── EMEAtrain_layer1_ID.conll
│       ├── EMEAdev_layer1_ID.conll
│       └── EMEAtest_layer1_ID.conll
├── preprocess_conll.py
├── ner_cnn_pretrained.py
├── train_ner_with_embeddings.py
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate conda environment:
```bash
conda create -n chatbot python=3.8
conda activate chatbot
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script with default parameters:
```bash
python train_ner_with_embeddings.py
```

Or specify custom parameters:
```bash
python train_ner_with_embeddings.py \
    --train QUAERO_FrenchMed/EMEA/EMEAtrain_layer1_ID.conll \
    --valid QUAERO_FrenchMed/EMEA/EMEAdev_layer1_ID.conll \
    --test QUAERO_FrenchMed/EMEA/EMEAtest_layer1_ID.conll \
    --embeddings model/word2vec_cbow_med_train20.model \
    --epochs 5
```

### Parameters

- `--train`: Path to training data (CONLL format)
- `--valid`: Path to validation data (CONLL format)
- `--test`: Path to test data (CONLL format)
- `--embeddings`: Path to pre-trained Word2Vec model
- `--epochs`: Number of training epochs (default: 5)

## Model Architecture

### CNN Model
- Pre-trained Word2Vec embeddings (100 dimensions)
- Multiple convolutional layers with different kernel sizes [3,4,5]
- Max pooling
- Dropout (0.2)
- Fully connected layer

## Output

Results are saved in `results.txt`, including:
- Training loss per epoch
- Classification metrics
  - Precision
  - Recall
  - F1-score
- Per-class performance

## Data Format

The CONLL files should have 5 columns:
1. Token index
2. Token
3. Unused
4. Unused
5. NER tag

Example:
```
1 EMEA 0 4 O
2 Epivir 47 53 B-CHEM
```

## Requirements

- Python 3.8
- PyTorch
- pandas
- scikit-learn
- transformers
- datasets
- accelerate==0.26.0

## Note

Make sure the Word2Vec model (`word2vec_cbow_med_train20.model`) is present in the `model/` directory before running the training script.