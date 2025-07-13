# FakeNewsDetector

Fine-tuning project for fake news detection using BERT and RoBERTa with HuggingFace and PyTorch. The goal was to explore how transformer models behave on the same task, not to build a production-ready system.

The fake news classification models was trained on the [LIAR2 dataset](https://huggingface.co/datasets/chengxuphd/liar2), which is an enhanced version of the original LIAR dataset by Wang (2017).


## 🧭 Project Overview

This prototype focuses on understanding how pre-trained models like `bert-base-uncased` and `roberta-base` perform on a multi-class fake news classification task using the LIAR v2 dataset. Both models were fine-tuned, evaluated, and compared side by side.

## ✅ What Was Done

### 1. Initial Setup and Testing
- Installed required dependencies.

### 2. Model Architecture Comparison
- Compared model structures and tokenizer behaviors.
- Observed training behavior and learning dynamics.

### 3. Dataset Preparation
- Used the `liar2` dataset (6-class version).

### 4. Fine-Tuning
- Trained each model with early stopping and saved the best checkpoints.
- Used validation loss to monitor training.

### 5. Metric-Based Evaluation
- Evaluated both models using accuracy, precision, recall, and F1 score.
- Visualized confusion matrices for deeper insight.

### 6. Comparison and Conclusion
- Results were decent but not outstanding.
- BERT and RoBERTa performed similarly overall.
- The project helped solidify understanding of Transformer fine-tuning in practice.

## 🛠️ Technologies

- PyTorch  
- HuggingFace Transformers & Datasets  
- scikit-learn  
- matplotlib / seaborn

---

This repo is an experiment and learning tool—not an optimized production system.
