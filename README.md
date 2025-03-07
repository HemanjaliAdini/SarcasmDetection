Enhanced Sarcasm Detection Using Hybrid CNN and Multi-Source Embeddings
![GitHub](https://github.com/HemanjaliAdini/SarcasmDetection)

## **Overview**
This project focuses on improving sarcasm detection in **online conversations** using a **hybrid CNN-BERT model** combined with **multi-source embeddings**. By integrating **personality traits, discourse-level embeddings, and content-based features**, the model aims to achieve **higher accuracy** in detecting sarcasm in social media conversations.

🔹 **Main Contributions:**
- **Hybrid Deep Learning Model** combining **CNN + BERT**
- **Multi-Source Embeddings:** Content-based, Personality, Discourse, and Stylometric features
- **Autoencoder-based Feature Fusion** for improved representation
- **Multi-Head Attention Mechanism** for contextual sarcasm detection
- **Extensive Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC

## **Dataset**
The model utilizes the **Self-Annotated Reddit Corpus (SARC)**, a **large-scale** dataset containing **sarcastic and non-sarcastic** comments with **conversation context**.

## **Model Architecture**
The sarcasm detection model is built using **deep learning components** designed for **context-aware detection**:
1. **Text Preprocessing & Embedding Generation**:
   - **BERT-based contextual embeddings**
   - **Stylometric, Personality, and Discourse embeddings**
   - **Feature Fusion using Autoencoders**
2. **Hybrid CNN + BERT Model**:
   - **BERT for contextual understanding**
   - **CNN for local pattern recognition (trigram features)**
   - **Multi-Head Attention Mechanism** for detecting sarcasm patterns
3. **Fully Connected Classification Layers**:
   - Dropout layers for regularization
   - Softmax activation for sarcasm classification

## **Results & Performance**
The **best-performing model** was the **Fusion Model (Personality + Content + Discourse + Stylometric embeddings)**.

📌 **Key Findings:**
- **Highest Accuracy:** **69.10%**
- **Best ROC-AUC Score:** **0.74**
- **Personality Embeddings Significantly Improve Performance**
- **Reduction in False Negatives** with context-aware embeddings

| Embedding Combination | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-----------------------|----------|-----------|--------|----------|---------|
| Stylometric + Content | 68.88% | 0.69 | 0.69 | 0.69 | 0.72 |
| Personality + Content | **69.24%** | 0.70 | 0.69 | 0.69 | **0.74** |
| Discourse + Content | 68.60% | 0.69 | 0.69 | 0.69 | 0.74 |
| **Fusion Model** | **69.10%** | 0.69 | **0.70** | **0.70** | **0.74** |
| Without Personality | 68.93% | 0.69 | 0.69 | 0.69 | 0.75 |

## **Installation & Setup**
### **1. Clone the Repository**
```bash
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
For inference on new text, use:

python
Copy
Edit
from sarcasm_detector import detect_sarcasm

text = "Oh great, another meeting at 8 AM. Just what I needed!"
prediction = detect_sarcasm(text)

print(f"Sarcasm Prediction: {prediction}")
Usage
💡 Applications of This Project:

Sarcasm Detection in Social Media: Identify sarcasm in comments and conversations.
Sentiment Analysis Enhancement: Improve sentiment classifiers with sarcasm detection.
Content Moderation: Reduce misinterpretations in online forums and automated moderation.
Future Improvements
🚀 Planned Enhancements:

Integration of Advanced Transformers: Exploring GPT and XLNet for improved sarcasm modeling.
Attention-Based Fusion Techniques: To improve performance across embedding types.
Personality-Aware Pretraining: Training models on personality-annotated datasets.
Generalization Across Domains: Testing on social media posts and news articles.
Contributors
👩‍💻 Hemanjali Adini

MSc Big Data Science, Queen Mary University of London
GitHub: HemanjaliAdini
Email: EC23629@qmul.ac.uk

