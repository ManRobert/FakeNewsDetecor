# FakeNewsDetector

Fine-tuning project for fake news detection using BERT and RoBERTa with TensorFlow and HuggingFace datasets.

## TODO – Project Phases

### 1. Initial Setup and Testing
- Install required dependencies.
- Run simple prediction using `bert-base-uncased` and `roberta-base`.

### 2. Model Architecture Comparison
- Analyze and compare the structure and training behavior of BERT vs RoBERTa.

### 3. Dataset Preparation
- Use the `liar2` dataset.
- Filter for binary labels (`true` / `false`).
- Explore basic augmentation techniques.

### 4. Fine-Tuning
- Tokenize inputs for both models.
- Create `tf.data` pipelines.
- Train models with adaptive learning rate and early stopping.

### 5. Metric-Based Evaluation
- Evaluate both models on the validation set.
- Compute accuracy, precision, recall, and F1 score.

### 6. Comparison and Conclusion
- Compare the performance of BERT and RoBERTa.
- Summarize key findings and provide practical recommendations.

## Technologies

- TensorFlow / Keras  
- HuggingFace Transformers  
- HuggingFace Datasets (`liar2`)  
- Scikit-learn for evaluation metrics
