# Speech Emotion Recognition (SER): A Comparative Analysis of Deep Learning Models & Loss Functions

##  Project Overview
This repository contains the implementation of a comprehensive comparative study on Speech Emotion Recognition (SER). The project evaluates the performance of four distinct feature extractors—**LSTM, VGGish, Wav2Vec 2.0, and HuBERT**—paired with five different loss functions to determine the optimal combinations for capturing emotional cues in speech.

All experiments were conducted on the benchmark **IEMOCAP dataset**, focusing on four emotion categories: anger, happiness, sadness, and neutral.

##  Repository Structure

The code is organized by model architecture. For **Wav2Vec 2.0**, specific notebooks are provided for each loss function investigated.

### Core Model Implementations
| File | Description |
| :--- | :--- |
| `HuBERT.ipynb` | Implementation of the **HuBERT** (Hidden-Unit BERT) model. In our study, this model achieved the highest overall accuracy (81.41%) when paired with Focal Loss. |
| `LSTM.ipynb` | Implementation of the **LSTM** (Long Short-Term Memory) network. This notebook explores sequential modeling of acoustic features. |
| `VGGish.ipynb` | Implementation of the **VGGish** CNN architecture, pre-trained on AudioSet, treating audio spectrograms as image inputs. |

### Wav2Vec 2.0 Loss Function Experiments
These notebooks contain the **Wav2Vec 2.0** implementation fine-tuned with specific objective functions:

| File | Loss Function Focus |
| :--- | :--- |
| `wav2vec2_cross_entropy.ipynb` | Standard **Categorical Cross-Entropy (CE)**, serving as the robust baseline. |
| `wav2vec2_focal_loss.ipynb` | **Focal Loss** implementation to address class imbalance and hard-to-classify samples. |
| `wav2vec2_label_smoothening_ce.ipynb` | **Label Smoothing**, used to prevent model overconfidence and improve generalization. |
| `wav2vec2_AAM_loss.ipynb` | **Additive Angular Margin (AAM)** loss, designed to enhance feature discriminability in the embedding space. |
| `wav2vec2_CCC_loss.ipynb` | **Concordance Correlation Coefficient (CCC)** loss, adapted to maximize agreement between predicted and actual emotion ratings. |

##  Experimental Results

The following table summarizes the accuracy (%) achieved by each model-loss combination on the IEMOCAP dataset:

| Model | Cross-Entropy | Label Smoothing | Focal Loss | AAM | CCC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LSTM** | 56.10 | 56.37 | 52.66 | 47.34 | **58.23** |
| **VGGish** | **57.54** | 57.00 | 56.19 | 27.73 | 55.65 |
| **Wav2Vec 2.0** | **74.46** | 72.17 | 72.29 | 61.33 | 74.22 |
| **HuBERT** | 81.30 | 80.09 | **81.41** | 78.66 | 63.80 |

**Key Findings:**
* **Best Overall:** HuBERT + Focal Loss achieved the state-of-the-art accuracy of **81.41%**.
* **Architecture Dependence:** Transformer-based models (HuBERT, Wav2Vec 2.0) significantly outperformed traditional LSTM and VGGish architectures.
* **Loss Function Synergy:** While Cross-Entropy is a strong baseline, **CCC Loss** proved most effective for the sequential LSTM model, whereas **Focal Loss** maximized the potential of the HuBERT transformer.

## ⚙️ Prerequisites
* Python 3.x
* PyTorch
* HuggingFace Transformers (for HuBERT and Wav2Vec 2.0)
* Librosa (for audio processing)
* IEMOCAP Dataset (Access must be requested from USC)

## 👥 Contributors
* Virti Rohit Mehta
* Samridhi Sahay
* Saumya Aryan
