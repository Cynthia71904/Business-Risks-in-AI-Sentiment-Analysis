# Overview

This project evaluates DistilBERT for sentiment analysis on Amazon Food Reviews. Despite 90% accuracy, the model fails on neutral reviews and mislabels some negative ones. KeyBERT keyword analysis shows limited contextual understanding, highlighting risks for business decisions. A Human-in-the-Loop workflow is proposed to mitigate these issues.

# Methods

Data: Amazon Food Reviews (1,000+ samples)

Preprocessing: Combine Summary + Text; map scores to Negative (1–2), Neutral (3), Positive (4–5); 80/20 train-test split

Model: DistilBERT fine-tuned for 3-class classification, AdamW optimizer, 2 epochs

Evaluation: Accuracy, confusion matrix, and keyword analysis with KeyBERT

# Key Findings

Neutral reviews often misclassified; some negative reviews labeled positive

Model fails to capture sarcasm and complex context

High accuracy can mislead business decisions

Mitigation: Human-in-the-Loop for ambiguous cases

# Usage

Place Reviews.csv in the project folder

Preprocess: python prepro.py

Train model: python sclassif.py

Analyze results: python reviews_anal.py

# License

MIT
