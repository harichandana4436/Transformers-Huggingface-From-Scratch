# Transformers-Huggingface-From-Scratch
Medium Blog link:https://medium.com/@tlakshmiharichandana123/transformers-and-hugging-face-a-complete-technical-blog-855e0847f943

# 📖 Overview

This repository accompanies a comprehensive technical blog on Transformer architecture and Hugging Face — covering everything from the limitations of RNNs/LSTMs to hands-on NLP implementations using AutoClasses.

# 📚 Table of Contents

1.Introduction to NLP

2.Limitations of RNNs and LSTMs

3.Why Transformers?

4.Transformer Architecture (Deep Dive)

5.Hugging Face — BERT Model Analysis

6.Practical Implementations

7.Results and Observations

8.Conclusion

9.Setup and Requirements

# 1. Introduction to NLP
Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language. It powers applications like:

🔍 Search Engines — understanding user intent

💬 Chatbots & Virtual Assistants — Siri, Alexa, ChatGPT

🌐 Machine Translation — Google Translate

📊 Sentiment Analysis — product review classification

📧 Spam Detection — email filtering

📝 Text Summarization — condensing long documents

# 2. Limitations of RNNs and LSTMs
Recurrent Neural Networks (RNNs)

RNNs process text word-by-word, left to right, carrying a hidden state as memory.
# Key Problems:
<img width="692" height="187" alt="image" src="https://github.com/user-attachments/assets/6059faed-388e-41a9-9d1c-6a62d77f5960" />

# Long Short-Term Memory (LSTMs)
LSTMs added three gates (Forget, Input, Output) to control information flow and partially address vanishing gradients — but remained sequential and slow for very long sequences.

# 3. Why Transformers?
The key insight: why process text sequentially at all?

Transformers process the entire sequence in parallel. Every word directly attends to every other word in a single step, regardless of distance — solving all three core RNN/LSTM problems.

         📄 Landmark Paper: "Attention Is All You Need" — Vaswani et al., Google Brain, June 2017

                             28.4 BLEU on EN→DE translation (new SOTA)
                             41.0 BLEU on EN→FR translation
                             Trained in 3.5 days on 8 GPUs
# 4. Transformer Architecture (Deep Dive)
## Architecture Overview
     INPUT SENTENCE

      ↓

    [ Input Embeddings + Positional Encoding ]

      ↓

    ┌─────────────────────────────┐
    │         ENCODER (×6)        │
    │  Multi-Head Self-Attention  │
    │  Add + LayerNorm            │
    │  Feed Forward Network       │
    │  Add + LayerNorm            │
    └─────────────────────────────┘

      ↓
    ┌─────────────────────────────┐
    │         DECODER (×6)        │
    │  Masked Self-Attention      │
    │  Add + LayerNorm            │
    │  Cross-Attention            │
    │  Add + LayerNorm            │
    │  Feed Forward Network       │
    │  Add + LayerNorm            │
    └─────────────────────────────┘
   
      ↓
    [ Linear + Softmax ] → OUTPUT TOKEN
## Components
### 2.1 Input Embeddings

Each token mapped to a dense vector of dimension d_model = 512. Semantically similar words cluster together in vector space.
### 2.2 Positional Encoding

Sine/cosine functions inject position information since Transformers have no inherent order:

                      PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
                      PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

### 2.3 Self-Attention (Q, K, V)

The core mechanism — every word attends to every other word:
 
                 Attention(Q, K, V) = softmax(QK^T / √d_k) × V

### 2.4 Multi-Head Attention

8 attention heads run in parallel, each capturing different relationship types (syntax, coreference, semantics, proximity, etc.), then concatenated and projected.
### 2.5 Feed Forward Network (FFN)
Applied position-wise to each token independently:

                FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂
                (512 → 2048 → 512 dimensions)
### 2.6 Residual Connections + Layer Normalization
                Output = LayerNorm(x + Sublayer(x))
                
Residuals provide gradient highways for stable deep training. Layer Norm normalizes across the feature dimension, independent of batch size.
### 5. Hugging Face — BERT Model Analysis
        Model: bert-base-uncased by Google Research
<img width="588" height="393" alt="image" src="https://github.com/user-attachments/assets/ea165b31-dbe5-4a21-853e-8da13cf31079" />

Training Data: BooksCorpus (800M words) + English Wikipedia (2.5B words)

Training Tasks: Masked Language Modeling (MLM) + Next Sentence Prediction (NSP)

Benchmark Results:
<img width="591" height="253" alt="image" src="https://github.com/user-attachments/assets/dc0e75af-6da7-4768-b89c-19bb89e47582" />
Intended Use: Classification, NER, QA, Semantic Similarity

Limitations: English-only, 512-token limit, inherits dataset biases, cannot generate text

### 6. Practical Implementations
#### Task 1 — Text Summarization

        Model: facebook/bart-large-cnn
BART combines a bidirectional encoder (like BERT) with an autoregressive decoder (like GPT), pre-trained with a denoising objective.

             📄 Code File: task_of_summarization_and_language_translation.ipynb
Output:

    Summary: AI is transforming industries from healthcare to finance.
    Machine learning models detect diseases with accuracy comparable to
    expert doctors. Researchers and policymakers are working to address
    concerns around job displacement, data privacy, and algorithmic bias.
#### Task-2
    Model: Helsinki-NLP/opus-mt-en-fr
MarianMT is a lightweight Transformer optimized for machine translation using shared SentencePiece tokenization across language pairs.
            
              📄 Code File:task_of_summarization_and_language_translation.ipynb

Output:

       English: Transformers have revolutionized the field of natural language processing.
       French : Les transformateurs ont révolutionné le domaine du traitement du langage naturel.

#### Model Comparison — BART vs MarianMT
<img width="767" height="293" alt="image" src="https://github.com/user-attachments/assets/e3476ad6-d59b-4d92-abe7-789bba64ccd3" />
### 7. Results and Observations
#### Summarization (BART)

✅ Produced fluent abstractive summaries (reformulated, not copied)

✅ Correctly preserved all key themes in ~52 words from a 90-word input

✅ Length control parameters (min_length, max_length) worked effectively

⚠️ Can hallucinate on out-of-domain text

⚠️ Large model size (~1.6 GB) requires significant GPU memory

#### Translation (MarianMT)

✅ Grammatically correct French output

✅ Correctly handled technical NLP vocabulary

✅ Applied French grammatical contractions naturally

⚠️ Struggles with idioms and informal language

⚠️ Limited to one language pair per model
### 8. Conclusion
The Transformer architecture replaced slow sequential processing with fast parallel self-attention, enabling models that are faster, smarter, and more scalable than anything before.

Three architectures explored in this project:
<img width="710" height="205" alt="image" src="https://github.com/user-attachments/assets/35f543a2-455c-4a19-806d-969dbfa1b67d" />
All are built on the same foundations: attention, FFN, residual connections, and layer normalization — introduced in a single 2017 paper that changed AI forever.

### 9. Setup and Requirements
Prerequisites
           
          bashpip install transformers torch sentencepiece
Running the Code
             
             bash# Clone the repo
             git clone https://github.com/<your-username>/<repo-name>.git
             cd <repo-name>
