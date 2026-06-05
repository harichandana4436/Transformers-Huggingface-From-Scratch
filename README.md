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

        Model: bart-large-cnn — available at `https://huggingface.co/facebook/bart-large-cnn`

**Architecture:** BART uses both the Encoder and Decoder stacks of the Transformer. The encoder is bidirectional (like BERT) and the decoder is autoregressive (like GPT). It is a denoising seq2seq model — corrupts input text and learns to reconstruct it.

**BART Architecture:**

**Input:** "The scientists discovered a new planet far from Earth."

**Step 1** — Tokenization + Positional Embedding:

        [ <s> ] [ The ] [ scientists ] [ discovered ] [ a ] [ new ] [ planet ] [ </s> ]
        ↓
        [ Token Embedding + Positional Embedding for each token ]
**Step 2** — Pass through 12 Encoder Layers (Bidirectional):
                 
        ┌──────────────────────────────────────────┐
        │            Encoder Layer 1               │
        │  Multi-Head Self-Attention (16 heads)    │
        │  (reads left AND right context)          │
        │  + Residual + LayerNorm                  │
        │  Feed Forward Network                    │
        │  + Residual + LayerNorm                  │
        └──────────────────────────────────────────┘
        ↓
        (Layers 2 to 12 repeat)
        ↓
        [ Encoder Hidden States — full context ]
**Step 3** — Pass through 12 Decoder Layers (Autoregressive):

       ┌──────────────────────────────────────────┐
       │            Decoder Layer 1               │
       │  Masked Multi-Head Self-Attention        │
       │  (can only see previous output tokens)   │
       │  + Residual + LayerNorm                  │
       │  Cross-Attention (attends to Encoder)    │
       │  + Residual + LayerNorm                  │
       │  Feed Forward Network                    │
       │  + Residual + LayerNorm                  │
       └──────────────────────────────────────────┘
       ↓
       (Layers 2 to 12 repeat)
       ↓
**Step 4** — Output (one token at a time):

         [ Astronomers ] [ found ] [ a ] [ new ] [ planet ] ...
         ↓
         Final summary string generated token by token
**Key specs:**
- 12 encoder layers + 12 decoder layers, 16 attention heads
 
- Hidden size: 1024, FFN dimension: 4096

- Total parameters: ~406 million

- Max sequence length: 1024 tokens

**Intended Use Cases:**

Text summarization (primary), translation, abstractive question answering, text generation. Not suitable for token-level tasks like NER or classification without modification.

**Training Data:**

Pre-trained on English language corpora (BookCorpus, CC-News, OpenWebText, Stories — ~160GB of text) using a denoising objective. Fine-tuned on CNN Daily Mail (over 300,000 news article–summary pairs) for abstractive summarization.

**Evaluation Metrics (on CNN DailyMail):**

          | Metric               | Score         |
          |----------------------|---------------|
          | ROUGE-1              | 42.949        | 
          | ROUGE-2              | 20.815        |
          | ROUGE-L              | 30.619        |
          | ROUGE-LSUM           | 40.038        |
          | Avg Generated Length | 78.587 tokens |
> State-of-the-art on CNN/DailyMail at time of release (2019)

**Limitations and Biases:**

- English-only

- 1024-token input limit (longer documents get truncated)

- Fine-tuned only on news articles — performs poorly on non-news domains

- May reflect CNN/DailyMail editorial biases

- Prone to hallucination (generating plausible but factually incorrect summary details)

- Not designed for conversational or dialogue text

**License:** MIT License — free for both commercial and non-commercial use.

## 📄 Code File
             📄 Code File: task_of_summarization_and_language_translation.ipynb
Output:

    Summary: AI is transforming industries from healthcare to finance.
    Machine learning models detect diseases with accuracy comparable to
    expert doctors. Researchers and policymakers are working to address
    concerns around job displacement, data privacy, and algorithmic bias.
#### Task-2
    Model: opus-mt-en-fr — available at `https://huggingface.co/Helsinki-NLP/opus-mt-en-fr`

**Architecture:** opus-mt-en-fr is a MarianMT model — a transformer encoder-decoder (seq2seq) with 6 layers in each component. It uses static sinusoidal positional embeddings (not learned) and does not have a layernorm embedding. It is trained using the Marian NMT framework and uses SentencePiece tokenization.

**MarianMT (opus-mt-en-fr) Architecture:**
**Input:** "The sun rises in the east."
**Step 1** — Tokenization with SentencePiece:

      [ <s> ] [ The ] [ sun ] [ rises ] [ in ] [ the ] [ east ] [ . ] [ </s> ]
      ↓
      [ Token Embedding + Static Sinusoidal Positional Embedding ]
      (positional values are fixed using sin/cos functions, not learned)
**Step 2** — Pass through 6 Encoder Layers (Bidirectional):
 
     ┌──────────────────────────────────────────┐
     │            Encoder Layer 1               │
     │  Multi-Head Self-Attention (8 heads)     │
     │  (reads full source sentence both ways)  │
     │  + Residual Connection                   │
     │  Feed Forward Network (dim: 2048)        │
     │  + Residual Connection                   │
     └──────────────────────────────────────────┘
     ↓
     (Layers 2 to 6 repeat)
     ↓
     [ Encoder Hidden States — source context ]
**Step 3** — Pass through 6 Decoder Layers (Autoregressive):

     ┌──────────────────────────────────────────┐
     │            Decoder Layer 1               │
     │  Masked Multi-Head Self-Attention        │
     │  (sees only previously generated tokens) │
     │  + Residual Connection                   │
     │  Cross-Attention (attends to Encoder)    │
     │  (aligns French output to English input) │
     │  + Residual Connection                   │
     │  Feed Forward Network (dim: 2048)        │
     │  + Residual Connection                   │
     └──────────────────────────────────────────┘
     ↓
     (Layers 2 to 6 repeat)
     ↓
**Step 4** — Output (one token at a time):

      [ Le ] [ soleil ] [ se ] [ lève ] [ à ] [ l' ] [ est ] [ . ]
      ↓
      Final French translation generated token by token

**Key specs:**

- 6 encoder layers + 6 decoder layers, 8 attention heads each

- Hidden size (d_model): 512, FFN dimension: 2048

- Vocabulary size: 59,514 tokens (SentencePiece)

- Dropout: 0.1, Activation: Swish

- Total parameters: ~74 million

**Intended Use Cases:**

English → French machine translation. Suitable for translating sentences and short paragraphs in news, general, and conversational domains. Part of the broader OPUS-MT project, which aims to make neural machine translation models widely available for many languages. Not designed for multilingual or multi-target translation without modification.

**Training Data:**

Trained on the OPUS dataset — a large open collection of parallel corpora. Pre-processing includes text normalization and SentencePiece subword tokenization. The model variant is `transformer-align`, released February 2020. OPUS aggregates data from sources like Europarl, OpenSubtitles, WikiMatrix, and UN corpus.

**Evaluation Metrics (BLEU scores on WMT/news test sets):**

| Test Set                  | BLEU | chrF   |
|---------------------------|------|--------|
| newsdiscusstest2015-enfr  | 40.0 | 0.643  |
| newsdiscussdev2015-enfr   | 33.8 | 0.602  |
| newstest2013              | 33.2 | 0.589  |
| newstest2011              | 34.3 | 0.611  |
| Tatoeba.en.fr             | 50.5 | 0.672  |

> BLEU measures n-gram overlap with reference translations; chrF measures character-level F-score.

**Limitations and Biases:**

- Trained on OPUS corpora which skews toward formal/news/subtitle text — performance may drop on technical, medical, or informal language

- Limited to English → French only (not reversible)

- SentencePiece vocabulary of ~59K tokens means rare words may be poorly handled

- May reflect gender and cultural biases present in the parallel training corpora (e.g., OpenSubtitles, Europarl)

- Max sequence length is 512 tokens — longer inputs get truncated

**License:** Apache 2.0 — free for both commercial and non-commercial use.

---

## Model Comparison

| Feature                 | BART (bart-large-cnn)  | MarianMT (opus-mt-en-fr) |
|-------------------------|------------------------|--------------------------|
| **Task**                | Summarization          | Translation (EN→FR)      |
| **Architecture**        | 12 Enc + 12 Dec layers | 6 Enc + 6 Dec layers     |
| **Parameters**          | ~406 million           | ~74 million              |
| **Hidden Size**         | 1024                   | 512                      |
| **Attention Heads**     | 16                     | 8                        |
| **Tokenizer**           | BPE                    | SentencePiece            |
| **Positional Encoding** | Learned                | Static Sinusoidal        |
| **Max Seq Length**      | 1024 tokens            | 512 tokens               |
| **Training Data**       | CNN/DailyMail (news)   | OPUS parallel corpus     |
| **Evaluation Metric**   | ROUGE                  | BLEU / chrF              |
| **License**             | MIT                    | Apache 2.0               |
| **Model Size**          | ~1.6 GB                | ~300 MB                  |

---

*Model card analyses written with reference to official Hugging Face model pages and the MarianMT documentation.*
 ## 📄 Code File
 
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
