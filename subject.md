# Semantics and Word Embeddings Course

## Session 1 - January 24th, 2025

### Course Material
- PDF: Introduction to distributed representations, recent approaches, evaluation approaches

### Lab Session 1 Instructions

#### Objective
Build and compare different word embedding approaches using gensim and fasttext libraries.

#### Resources
- [Word2vec Documentation](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec)
- [Fasttext Documentation](https://fasttext.cc/docs/en/support.html)
- [Corpus Download](https://perso.limsi.fr/neveol/TP_ISD2020.zip)

#### Data
- QUAERO_FrenchMed_traindev.ospl
- QUAERO_FrenchPress_traindev.ospl
(One sentence per line format, tokens separated by spaces)

#### Requirements
- Use Google Colab
- Install gensim and fasttext

#### Tasks

1. **Word Embeddings Training**
   - Create Python and bash scripts for training:
     - word2vec (Cbow, skipgram)
     - fasttext (Cbow)
   - Train on both medical and non-medical corpora
   - Hyperparameters:
     - dim = 100
     - min_count = 1

2. **Semantic Similarity**
   - Find closest words using cosine similarity
   - Evaluation comparisons:
     - Same corpus, different embeddings
     - Same approach, different corpora
   - Test words: `patient, traitement, maladie, solution, jaune`
   - Methods:
     - scipy's `spatial` method
     - gensim's `most_similar` method

## Session 2 - February 7th, 2025

### Lab Session 2 Instructions

#### Objective
Evaluate embeddings performance on Named Entity Recognition (NER) in medical domain.

#### Resources
- Scripts for sequence classification (LSTM, CNN, BERT)
- Same corpus as Session 1

#### Data Format
- Conll-like format
- 5 columns: token index, token, 2 unused columns, NER tag
- Tags: I-TYPE, B-TYPE, O

#### Tasks

1. **NER Model Training**
   - Adapt scripts for token classification
   - Modify CNN script to use TP1 embeddings
   - Update transformer script for token classification
   - Evaluate using recall, precision, and F1

#### Analysis Questions
1. Which model performs best on each dataset?
2. What are the differences between embeddings from different corpora?
3. How do results compare with Transformer model?

### Final Report Requirements
Due: February 14th, 2025
- Summarize both lab sessions
- Include semantic similarity analysis
- Include NER results and question responses