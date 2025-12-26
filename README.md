# LSTM Predict Next Word 🧠🔤

A PyTorch-based **Next Word Prediction** project using a Long Short-Term Memory (LSTM) network.  
This repository demonstrates how to build, train, and evaluate an LSTM model that predicts the next word given a sequence of text. You’ll find data preprocessing, model architecture, training, evaluation, and example predictions in a step-by-step notebook and Python scripts.

---

## 📌 Project Overview

This project implements a simple language model that learns patterns in text and predicts the most probable next word given a sequence of previous words.

✔ Text preprocessing & tokenization  
✔ Sequence generation for training  
✔ LSTM neural network in PyTorch  
✔ Training & evaluation loops  
✔ Prediction examples

The walkthrough is available in both notebook and script formats.

---

## 🧠 Why Next Word Prediction?

Next Word Prediction is a foundational NLP task where a model learns language patterns — useful in:
- Autocomplete systems
- Text suggestion features
- Basic language modeling

LSTM networks are well-suited for sequence learning because they can capture long-term dependencies in text data, a key requirement for this task. :contentReference[oaicite:0]{index=0}

---

## Model Prediction
```aiignore
input: ['ghost', 'do', 'not', 'forget', 'this']
Model prediction: visitation 
True prediction: visitation
```
## ⚠️ Model Status & Training Behavior

**Current Status:** Overfitted (Work in Progress)

The current LSTM model successfully learns patterns from the training data but shows signs of **overfitting** when evaluated on unseen data. This behavior is expected at this stage due to factors such as a limited dataset size, large vocabulary, and training the model from scratch.

Overfitting here confirms that:
- ✅ The data preprocessing and pipeline are correct  
- ✅ The LSTM architecture is functioning as intended  
- ✅ Gradients, loss computation, and optimization are working  

The next steps focus on improving **generalization**, including:
- Reducing and tuning vocabulary size  
- Controlling sequence length  
- Adding regularization techniques (dropout)  
- Hyperparameter tuning and experimentation  

### 📉 Training vs Test Loss

The plot below shows the comparison between **training** and **testing**, clearly highlighting the current overfitting behavior:

<img src="plt" alt="Training vs Test Loss" width="500"/>

> This visualization helps track learning progress and guides further improvements to balance training performance and generalization.

© 2025 dctn  
This repository is shared for learning purposes.  
Unauthorized use without attribution is discouraged.
