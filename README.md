# FakeNewsDetector

BERT-based fine-tuning project for fake news detection using TensorFlow and HuggingFace datasets.

## TODO – Project Phases

### 1. initial setup and testing
install required libraries, run simple prediction with pretrained model.

### 2. pretrained model research
understand architecture (`bert-base-uncased`), layers, training behavior.

### 3. dataset research for fine-tuning and augmentation
select dataset and filter true/false labels. explore possible augmentation and experiment.

### 4. implement fine-tuning and augmentation
tokenization, tf.data pipeline, compile and train with adaptive learning rate + early stopping.

### 5. metric-based evaluation
compute accuracy, precision, recall, and f1 score on validation set.

### 6. final evaluation and contribution
summarize model performance, impact of fine-tuning, and project insights.

## Technologies
- tensorflow / keras  
- huggingface transformers  
- datasets: liar2  
- sklearn for metrics  
