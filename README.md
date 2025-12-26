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
