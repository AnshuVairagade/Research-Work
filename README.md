# Speaker-Aware NLP for Diarized Conversations

## 📌 Project Overview

This project integrates **Speaker Diarization**, **Speaker-Aware BERT
Fine-Tuning**, and **Novel Evaluation Metrics** to enhance understanding
of multi-speaker conversations.

-   **Part 1: Speaker Diarization**
    -   Separates audio conversations into speaker-specific segments.
    -   Uses **ECAPA-TDNN embeddings** for robust speaker
        representation.
    -   Clustering methods: Spectral Clustering, Improved K-Means, and
        U-K-Means.
-   **Part 2: Speaker-Aware BERT Fine-Tuning**
    -   Extends vanilla BERT by incorporating **speaker identity** and
        **turn position embeddings**.
    -   Helps the model learn not just *what is said*, but also *who
        said it* and *when*.
-   **Part 3: Novel Metrics**
    -   Beyond standard accuracy/F1, we introduce
        **speaker-context-aware metrics** that evaluate performance on
        multi-turn, multi-speaker conversations.

------------------------------------------------------------------------

## 🚀 Features

-   Speaker diarization pipeline with embeddings + clustering.
-   Speaker-aware embeddings integration into BERT.
-   Automated labeling with HuggingFace sentiment/emotion models.
-   Training/evaluation scripts with visualization tools.
-   Custom metrics to evaluate model performance in conversational AI
    tasks.

------------------------------------------------------------------------

## 📂 Project Structure

 <img width="443" height="651" alt="image" src="https://github.com/user-attachments/assets/4ddff686-007a-46f5-80ee-63719e83f41f" />


------------------------------------------------------------------------

## ⚙️ Installation

``` bash
# Clone repository
git clone https://github.com/username/speaker-aware-nlp.git
cd speaker-aware-nlp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venvScriptsactivate     # Windows

# Install dependencies
pip install -r requirements.txt
```

------------------------------------------------------------------------

## 📊 Methodology

### 🔹 1. Speaker Diarization

-   Extract embeddings using **ECAPA-TDNN (SpeechBrain)**.
-   Normalize embeddings and build affinity matrices.
-   Apply clustering:
    -   Spectral Clustering
    -   Improved K-Means
    -   U-K-Means
-   Visualize results using **t-SNE** and **heatmaps**.

### 🔹 2. Speaker-Aware BERT Fine-Tuning

-   Extend BERT with:
    -   **Speaker embeddings** (who is speaking).
    -   **Turn position embeddings** (order in dialogue).
-   Fine-tune on diarized and labeled conversational datasets.
-   Evaluate with classification tasks (sentiment, intent, etc.).

### 🔹 3. Novel Metrics

-   Evaluate beyond accuracy:
    -   Speaker-contextual accuracy.
    -   Turn-aware precision/recall.
    -   Dialogue-level consistency score.

------------------------------------------------------------------------

## 🧪 Experiments

-   **Dataset:** Conversational datasets with real-world audio + text.
-   **Evaluation:**
    -   Diarization → DER (Diarization Error Rate), clustering
        accuracy.
    -   NLP model → Accuracy, F1-score, Confusion Matrix.
    -   Novel metrics → Speaker-context-aware evaluation.
-   **Visualization:**
    -   Training curves (loss, accuracy).
    -   Embedding clusters (t-SNE, PCA).
    -   Confusion matrices.

------------------------------------------------------------------------

## 📈 Results

-   **Speaker Diarization:**
    -   Improved clustering methods showed lower DER compared to vanilla
        k-means.
-   **Speaker-Aware BERT:**
    -   Outperformed vanilla BERT in capturing multi-speaker context.
-   **Novel Metrics:**
    -   Provided deeper insights into model performance for
        dialogue-based tasks.

------------------------------------------------------------------------

## 🛠️ Usage

### 1. Run Speaker Diarization

``` bash
python diarization/run_pipeline.py --audio_file sample.wav
```

### 2. Train Speaker-Aware BERT

``` bash
python nlp/train.py --config configs/train_config.json
```

### 3. Evaluate Model

``` bash
python nlp/evaluate.py --checkpoint model.ckpt
```

### 4. Compute Novel Metrics

``` bash
python metrics/speaker_context_score.py --predictions preds.json --labels labels.json
```

------------------------------------------------------------------------

## 🔮 Future Work

-   End-to-end pipeline from **raw audio → diarization → NLP →
    evaluation**.
-   Apply to multi-lingual and noisy real-world datasets.
-   Explore LLM-based speaker-aware architectures.

------------------------------------------------------------------------

## 👨‍💻 Contributors

-   **Anshu Vairagade** -- Research, Implementation, Paper Writing
-   Open-source tools: SpeechBrain, HuggingFace, PyTorch

------------------------------------------------------------------------

## 📜 Citation

If you use this project in your research, please cite:

    @inproceedings{vairagade2025speakeraware,
      title={Speaker-Aware NLP with Diarization and Novel Metrics},
      author={Anshu Vairagade},
      booktitle={IEEE CIACON 2025},
      year={2025}
    }
