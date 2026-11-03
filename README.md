# Bearing fault diagnosis with a fine-tuned LLM and retrieval

This repository holds the code behind my master's thesis at AIMS Rwanda, supervised by Prof. Vishnu S. Pendyala (San José State University). It is also the code of a paper published in *Machine Learning with Applications* (2026): [10.1016/j.mlwa.2026.101057](https://doi.org/10.1016/j.mlwa.2026.101057).

The system reads a window of vibration signal from a bearing, turns 21 features of that signal into a short text, and asks a fine-tuned Llama 3.2 3B model for a structured report: fault class, severity, an explanation that quotes the signal features, and a maintenance recommendation. A retrieval step (FAISS) adds similar past cases to the prompt.

I tested it on two public datasets, CWRU and Paderborn. On classification alone, the LLM is less accurate than an SVM or a Random Forest trained on the same features. Retrieval helps it a lot in the cross-load test, but the results change a lot from one training seed to another. I did not check whether technicians find the reports useful.

## How it works

1. Each recording is cut into windows of 16,384 samples with 75% overlap.
2. 21 features are computed per window, in the time and frequency domains. The FFT uses a Hann window, and some features are the signal energy in bands around the expected fault frequencies.
3. The features, with the operating conditions, are written as key-value text.
4. The 3 most similar training windows are retrieved with FAISS (`all-MiniLM-L6-v2` embeddings) and added to the prompt.
5. Llama 3.2 3B Instruct is fine-tuned with LoRA on a 4-bit NF4 base model (r=16, alpha=32, 24.31M trainable parameters, 0.75% of the total). The loss is computed on the report only, not on the prompt.
6. The report is generated with greedy decoding, and the fault class is parsed from the text.

## Results

Macro-F1 (%) on held-out test windows. SVM and Random Forest use the same 21 features and the same recording-level splits as the LLM.

| Experiment | SVM | Random Forest | LLM |
| :--- | :---: | :---: | :---: |
| CWRU, in-distribution (976 test windows, one run) | 93.15 | 91.21 | 78.45 |
| Cross-load, no retrieval (seed 42) | 87.48 | 89.42 | 19.91 |
| Cross-load, with retrieval (seed 42) | 87.48 | 89.42 | 67.89 |
| Cross-load, no retrieval (5 seeds, mean ± SD) | 87.48 | 89.42 | 45.38 ± 15.25 |
| Cross-load, with retrieval (5 seeds, mean ± SD) | 87.48 | 89.42 | 66.01 ± 8.19 |
| Paderborn, CWRU model without retraining | – | – | 21.02 |
| Paderborn, after fine-tuning on Paderborn | 63.90 | 80.39 | 77.33 |

In the cross-load experiment, the LoRA adapter, the retrieval index and the SVM and Random Forest tuning all use only the 0–1 HP recordings. Every model is tested on the same 2,308 windows from the 2–3 HP recordings. A small InceptionTime network trained on the raw signal, with five seeds, reaches 84.04 ± 6.29.

Retrieval improved the LLM in every one of the five seeds (seeds 11, 22, 33, 44 and 55), by 12.3 to 40.2 points. The paired t-test gives p = 0.015. The Wilcoxon test gives p = 0.0625, which is the smallest value possible with five pairs. Because the seed-to-seed spread is large, the seed-42 numbers (19.91 and 67.89) should not be read as typical.

Older versions of this README showed numbers from the first version of the experiments. The table above replaces them with the results of the published paper.

## What these results do not show

- The jump from 21.02 to 77.33 on Paderborn comes from training a new adapter and a new retrieval index on Paderborn data. It is adaptation to a new domain. It does not show that one fixed model generalizes across datasets.
- The Paderborn split separates recordings but not bearings. Every bearing in the validation and test sets also appears in training, so Paderborn scores may be too optimistic. The Paderborn severity labels are a placeholder (all "Medium"), so severity is not evaluated there.
- A numerical check on the explanations shows that they copy the right band energies and frequencies. It does not show that the diagnosis or the maintenance advice is correct, and no expert reviewed them.
- Readability scores (`textstat`) measure the complexity of the language only. They say nothing about whether the reports are understandable.
- I could not show that the operating metadata or the expected-frequency text improves accuracy. These ablations used one seed, and the pattern was not consistent.

## Files

- `main_notebook.py`: a Jupyter notebook exported as a Python script, to run cell by cell. It covers CWRU feature extraction, the text encoding, the FAISS index, LoRA training, the SVM and Random Forest baselines, the cross-load experiment, the Paderborn experiments, and the checks added for the paper revision (five seeds, retrieval depth, sentence encoders, input ablations, numerical consistency).
- `lora_utils.py`: the LoRA and quantization configuration, the loss mask for the report, and the training function.
- `train_worker.py`: trains one adapter in its own process, so that each seed runs on one GPU.

The last three cells of the notebook (the InceptionTime baseline, the Paderborn split audit and the reproducibility manifest) are commented out in this file. Uncomment them to run them.

## Running it

The data is not in this repository. Download the 48 kHz drive-end files of the [CWRU bearing data](https://engineering.case.edu/bearingdatacenter/download-data-file) (40 `.mat` files) and the Paderborn University bearing dataset (Lessmeier et al., PHM Europe, 2016). Llama 3.2 needs access from Meta on Hugging Face and a login.

The paths and the GPU choice are set for my machine. Before running, change `base_dir` and `CUDA_VISIBLE_DEVICES` near the top of `main_notebook.py`, and `PROJECT_DIR` in the validation-suite cell. CWRU files go in `<base_dir>/cwru_data` and Paderborn files in `<base_dir>/paderborn_data`. `train_worker.py` must be in `PROJECT_DIR`.

Each training run used one NVIDIA RTX 3090 (24 GB) on the AIMS HPC cluster. Versions used:

- Python 3.11+ and PyTorch 2.x
- `transformers` 4.47+, `peft` 0.14+, `trl` 0.13+
- `bitsandbytes` 0.45.5+
- `sentence-transformers` 3.3+, `faiss-cpu` 1.9+
- `scikit-learn` 1.4+ (scaling, grouped cross-validation, SVM and Random Forest)
- `textstat` 0.7+

## Citation

Danbe, J. S., and Pendyala, V. S. (2026). RAG-Augmented LoRA Fine-Tuned LLM for Structured Bearing Fault Diagnosis in Industry 4.0 Predictive Maintenance. *Machine Learning with Applications*. https://doi.org/10.1016/j.mlwa.2026.101057

```bibtex
@article{danbe2026bearing,
  author  = {Danbe, Joachim Sinyabe and Pendyala, Vishnu S.},
  title   = {RAG-Augmented LoRA Fine-Tuned LLM for Structured Bearing Fault Diagnosis in Industry 4.0 Predictive Maintenance},
  journal = {Machine Learning with Applications},
  year    = {2026},
  doi     = {10.1016/j.mlwa.2026.101057}
}
```

Portfolio: https://yehoyakim.netlify.app
