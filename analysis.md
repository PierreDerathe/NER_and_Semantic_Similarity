# Word Embeddings and Semantic Similarity Analysis

## 1. Experimental Setup

We trained different word embedding models on medical and non-medical corpora:

- Word2Vec models (CBOW and Skipgram)
- FastText models (CBOW)

Parameters used:
- Vector dimension: 100
- Minimum word count: 1 
- Context window: 5
- Training epochs: 20 (after experimentation)

## 2. Model Optimization

We first experimented with different numbers of epochs (1, 5, 20, 100) for the CBOW model on medical corpus. Analysis showed:
- 1 epoch: Poor semantic relationships, mostly syntactic similarities
- 5 epochs: Better relationships but still superficial
- 20 epochs: Good balance of semantic relationships and training time
- 100 epochs: Slightly better results but diminishing returns vs computational cost

We chose 20 epochs as the optimal setting for subsequent training.

## 3. Comparative Analysis

### 3.1 Impact of Embedding Approaches (Same Corpus)

#### Medical Corpus:

1. **"patient"**:
- Word2Vec CBOW: Focuses on medical context ("carte", "aptitude", "risque")
- Word2Vec Skipgram: Similar but more specific medical terms ("allergique", "examiner")
- FastText: Better handles morphological variants ("Patient", "patiente")

2. **"traitement"**:
- Word2Vec CBOW: Medical process-oriented ("début", "Parkinson", "VIH")
- Word2Vec Skipgram: More temporal aspects ("premiers", "début", "suivi")
- FastText: Captures morphological variations ("Traitements", "ajustement")

3. **"maladie"**:
- Word2Vec CBOW: Specific conditions ("infection", "Parkinson")
- Word2Vec Skipgram: More disease types ("chronique", "leucémie")
- FastText: Includes related medical terms ("souffrance", "somatostatine")

### 3.2 Impact of Corpus Domain (Same Approach)

1. **Medical vs Non-Medical (Word2Vec CBOW)**:
- Medical corpus: More precise medical terminology
- Non-medical corpus: General usage and contextual meanings

Example "traitement":
- Medical: Medical procedures and conditions
- Non-medical: Administrative/general concepts ("coût", "système", "financement")

2. **Domain Specificity**:
- Technical terms show stronger domain alignment
- Common words show different semantic associations per domain

## 4. Key Findings

1. **Model Characteristics**:
- Word2Vec CBOW: Good for general domain relationships
- Word2Vec Skipgram: Better for specific technical terms
- FastText: Superior for morphological variations and rare words

2. **Corpus Impact**:
- Medical corpus produces more specialized, technical relationships
- Non-medical corpus reflects everyday usage and broader contexts

3. **Word Type Influence**:
- Technical medical terms: Strong domain-specific relationships
- Common words: Different but valid semantic spaces in each domain

## 5. Limitations and Considerations

1. **Corpus Size**:
- Medical corpus is smaller, potentially affecting model robustness
- Non-medical corpus provides broader but less specialized coverage

2. **Model Behavior**:
- FastText shows advantage with morphologically rich words
- Word2Vec models might miss rare word variations

## 6. Conclusions

The analysis demonstrates that:
1. Domain-specific training is crucial for specialized applications
2. FastText offers advantages for morphological variation
3. Choice of embedding approach should consider specific use case needs
4. Corpus domain significantly impacts semantic relationships

Future work could explore:
- Combining domain-specific and general embeddings
- Testing on specific medical NLP tasks
- Evaluating with larger medical corpora

## 7. NER Task Performance Analysis

### 7.1 Model Performance Overview

1. **Class Imbalance Issue**
- Majority class ('O'): 12,141 samples (79% of dataset)
- Minority classes: severely underrepresented (e.g., I-DEVI: 1 sample)

2. **Training Evolution**
- Word2Vec CBOW:
  * Epoch 1: Shows some learning (f1-score: 0.51)
  * Epoch 3: Converges to majority class (f1-score: 0.70)
- Word2Vec Skipgram:
  * Similar pattern but slightly worse initial performance
  * Final convergence to majority class prediction

3. **Performance Metrics**
- Final accuracy (0.79) is misleading due to class imbalance
- Zero performance on minority classes
- Perfect recall (1.00) for 'O' class indicates model bias

### 7.2 Key Issues Identified

1. **Severe Class Imbalance**
- Medical entity tags (B-*, I-*) are underrepresented
- Model optimizes for majority class ('O')

2. **Model Limitations**
- Simple CNN architecture struggles with sequence labeling
- Loss function doesn't account for class weights
- No specific handling of entity boundaries (B-* vs I-*)

### 7.3 Recommended Improvements

1. **Data-level Solutions**
- Implement class weighting in loss function
- Consider data augmentation for minority classes
- Explore oversampling techniques for medical entities

2. **Model Architecture**
- Add CRF layer for better sequence modeling
- Implement attention mechanisms
- Consider hierarchical classification approach

3. **Training Strategy**
- Use focal loss or weighted cross-entropy
- Implement curriculum learning
- Consider multi-task learning approaches

These results suggest that while word embeddings capture semantic relationships well, additional techniques are needed for effective NER in medical texts.

