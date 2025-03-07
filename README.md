Sarcasm Detection Using Hybrid CNN and Multi-Source Embeddings


Overview
This project presents an advanced sarcasm detection model leveraging Hybrid CNN and Multi-Source Embeddings. By integrating personality traits, discourse-level embeddings, and content-based features with a BERT-based model and Convolutional Neural Network (CNN) architecture, the approach aims to significantly improve sarcasm detection accuracy in social media conversations.

The model processes data from the Self-Annotated Reddit Corpus (SARC) and applies deep learning paradigms with multiple embedding types to capture the nuances of sarcasm effectively.

Features
Hybrid CNN + BERT Model for sarcasm detection
Multi-Source Embeddings:
Content embeddings
Personality embeddings
Discourse embeddings
Stylometric embeddings
Autoencoder for Feature Fusion to improve representation
Multi-Head Attention Mechanism for context-aware sarcasm detection
Evaluation Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC
Dataset
The model utilizes the Self-Annotated Reddit Corpus (SARC), a large dataset containing naturally occurring sarcastic and non-sarcastic comments from Reddit along with conversational context.

Architecture
The sarcasm detection system consists of:

Text Preprocessing:
Tokenization, lowercasing, punctuation removal
Embedding generation using BERT and BERTopic models
Feature fusion using Autoencoders
BERT for Feature Extraction:
Contextual embeddings of sentences
Fine-tuning BERT for sarcasm detection
Convolutional Neural Network (CNN):
Captures local dependencies (trigram patterns, n-grams)
Feature extraction via Conv1D, ReLU activation, and max pooling
Multi-Head Attention Mechanism:
Enhances contextual understanding across long text sequences
Helps detect sarcasm patterns within conversational context
Classification Layer:
Fully connected layers for classification
Dropout for regularization
Softmax activation for sarcasm prediction
Results
Best Performing Model: Fusion Model (Personality + Content + Discourse + Stylometric)
Accuracy: 69.10%
ROC-AUC Score: 0.74
Reduction in False Negatives when using personality embeddings
Improved Precision-Recall Balance with feature fusion
Installation & Requirements
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/HemanjaliAdini/SarcasmDetection.git
cd SarcasmDetection
2. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Run the Model
To train and evaluate the sarcasm detection model:

bash
Copy
Edit
python train.py
For inference on new text:

python
Copy
Edit
from sarcasm_detector import detect_sarcasm
text = "Oh great, another meeting at 8 AM. Just what I needed!"
prediction = detect_sarcasm(text)
print(f"Sarcasm Prediction: {prediction}")
Usage
Sarcasm Detection in Social Media: Helps identify sarcasm in comments and conversations.
Sentiment Analysis Enhancement: Improves the performance of sentiment classifiers.
Content Moderation: Detects sarcasm in online forums and reduces misinterpretations.
Future Work
Incorporating transformers like GPT or XLNet for better sarcasm modeling
Implementing Personality-Aware Pretraining to enhance sarcasm detection
Expanding dataset sources beyond Reddit for broader applicability
Explainability & Interpretability: Improving the transparency of sarcasm classification
Contributors
👩‍💻 Hemanjali Adini

MSc Big Data Science, Queen Mary University of London
GitHub: HemanjaliAdini
Email: EC23629@qmul.ac.uk
License
This project is licensed under the MIT License - see the LICENSE file for details.
