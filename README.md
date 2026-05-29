# AI Assistant for Industrial Bearing Fault Diagnosis using LLMs and RAG

An end-to-end predictive maintenance diagnostic pipeline designed in the context of Industry 4.0. This repository contains the code and configuration to map raw vibration signals into structured, human-readable natural language diagnostic reports using a fine-tuned **LLaMA 3.2 3B Instruct** model combined with **Retrieval-Augmented Generation (RAG)**.

The framework is evaluated on two prominent benchmark datasets: the **Case Western Reserve University (CWRU)** and the **Paderborn University** bearing datasets.

---

## Key Features

* **Advanced Signal Processing:** 21 engineered features extracted from both Time and Frequency domains (FFT resolution optimization, Hann-windowing, and dedicated band-energy tracking).
* **Sensor-to-Text Statistical Encoding:** Converts high-dimensional numerical feature vectors into structured key-value descriptive text prompts readable by LLMs.
* **Inference-Time Adaptability via RAG:** Leverages an exact nearest-neighbor vector index (`faiss-cpu`) with `all-MiniLM-L6-v2` dense embeddings to prepend historical cases as few-shot context, bridging the generalization gap under intra-dataset distribution shifts.
* **Parameter-Efficient Fine-Tuning (PEFT):** Uses 4-bit NF4 quantization (`bitsandbytes`) and **LoRA** ($r=16, \alpha=32$) to fine-tune only 0.75% (24.31M) of total model weights.
* **Completion-Only Loss Masking:** Implements custom chat-template masking during Supervised Fine-Tuning (SFT) to avoid prompt memorization and guide accurate autoregressive loss computation.

---

## Consolidated Experimental Results

The pipeline has been thoroughly verified across 4 core industrial scenarios (Macro F1 and Accuracy):

| Exp. | Scenario | Best Classical Baseline | Proposed LLM System | Key Takeaway |
| :--- | :--- | :---: | :---: | :--- |
| **1** | CWRU In-Distribution | **95.07% F1** (SVM) | 87.74% F1 | Classical models maintain an accuracy edge but lack explanation capability. |
| **2** | CWRU Cross-Load (0-1 -> 2-3 HP) | 87.03% F1 (SVM) | **91.13% F1** (with RAG) | **RAG yields a +3.00 pp gain**, enabling the LLM to outperform classical boundaries. |
| **3** | Zero-Shot Transfer (CWRU -> Pad) | - | 23.65% F1 | Severe domain shift renders zero-shot deployment unviable. |
| **4** | Paderborn Re-Fine-Tuned | **80.12% F1** (RF) | 78.91% F1 | Re-fine-tuning recovers **+55.26 pp**, confirming architectural portability. |

---

## Software Stack & Environment

All experiments were executed on an HPC node equipped with **NVIDIA RTX 3090 (24GB VRAM)** GPUs. 

Dependencies are pinned to guarantee reproducible executions:
* `Python` 3.11+
* `PyTorch` 2.x
* `transformers` 4.47+ & `peft` 0.14+ (LLM initialization and LoRA scaling)
* `trl` 0.13+ (`SFTTrainer` instruction execution environment)
* `bitsandbytes` 0.45.5+ (4-bit double-quantization configuration)
* `sentence-transformers` 3.3+ & `faiss-cpu` 1.9+ (Vector database mechanics)
* `scikit-learn` 1.4+ (Standard scaling, GroupKFold validation, SVM & RF baselines)
* `textstat` 0.7+ (Readability indexes computation: Flesch Reading Ease, Flesch-Kincaid Grade)

---

## Getting Started

### Installation

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/YOUR_USERNAME/bearing-fault-diagnosis-llm.git](https://github.com/YOUR_USERNAME/bearing-fault-diagnosis-llm.git)
cd bearing-fault-diagnosis-llm
pip install -r requirements.txt
