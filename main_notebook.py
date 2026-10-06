#!/usr/bin/env python
# coding: utf-8

# # Bearing Fault Diagnosis with CWRU and Paderborn Datasets


# ## Environment and Dependencies

# In[1]:


# !pip uninstall -y bitsandbytes peft trl accelerate transformers \
#     sentence-transformers datasets faiss-cpu huggingface_hub -q


# !pip install \
#     "bitsandbytes>=0.45.5" \
#     "transformers>=4.47.0" \
#     "accelerate>=1.2.0" \
#     "peft>=0.14.0" 
\
#     "trl>=0.13.0" \
#     "sentence-transformers>=3.3.0" \
#     "datasets>=3.2.0" \
#     "faiss-cpu>=1.9.0" \
#     "huggingface_hub>=0.26.0" \
#     "fsspec>=2025.3.0" \
#     --quiet


# In[2]:

from IPython.display import display
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# In[3]:


import torch
print("GPUs visibles:", torch.cuda.device_count())


# In[4]:


# import sys, torch
# print("Python:", sys.executable)
# print("PyTorch:", torch.__version__)
# print("CUDA:", torch.cuda.is_available())
# for i in range(torch.cuda.device_count()):
#     mem_free = torch.cuda.mem_get_info(i)[0] / 1e9
#     mem_total = torch.cuda.get_device_properties(i).total_memory / 1e9
#     print(f"  GPU {i}: {torch.cuda.get_device_name(i)} | Libre: {mem_free:.1f}/{mem_total:.1f} GB")


# ## Project Paths

# In[5]:


from pathlib import Path


base_dir = Path('/home/yehoyakim/2026/Data')

data_dir = base_dir / 'cwru_data'
output_dir = base_dir / 'outputs'
fig_dir = output_dir / 'figures'

# for d in [output_dir, fig_dir]:
#      d.mkdir(exist_ok=True, parents=True)

mat_files = list(data_dir.glob('*.mat'))
print(f'{len(mat_files)} .mat files found in {data_dir}')
if not mat_files:
    print('No CWRU .mat files were found. Please check data_dir.')


# ## Imports

# In[6]:


import os
import gc
import re
import json
import pickle
import warnings
from collections import Counter
import random

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.fft import rfft, rfftfreq
from scipy.stats import kurtosis, skew

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import faiss
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel
)

from trl import SFTTrainer, SFTConfig
from tqdm.auto import tqdm

from transformers import DataCollatorForLanguageModeling

from sklearn.model_selection import GroupKFold, ParameterGrid


warnings.filterwarnings('ignore')


# ## CWRU File Map and Label Mapping

# In[7]:


file_map = {
    # Normal
    97:("Normal",0.000), 98:("Normal",0.000),
    99:("Normal",0.000), 100:("Normal",0.000),
    # Inner Race 0.007"
    109:("Inner Race",0.007), 110:("Inner Race",0.007),
    111:("Inner Race",0.007), 112:("Inner Race",0.007),
    # Ball 0.007"
    122:("Ball",0.007), 123:("Ball",0.007),
    124:("Ball",0.007), 125:("Ball",0.007),
    # Outer Race 0.007"
    135:("Outer Race",0.007), 136:("Outer Race",0.007),
    137:("Outer Race",0.007), 138:("Outer Race",0.007),
    # Inner Race 0.014"
    174:("Inner Race",0.014), 175:("Inner Race",0.014),
    176:("Inner Race",0.014), 177:("Inner Race",0.014),
    # Ball 0.014"
    189:("Ball",0.014), 190:("Ball",0.014),
    191:("Ball",0.014), 192:("Ball",0.014),
    # Outer Race 0.014"
    201:("Outer Race",0.014), 202:("Outer Race",0.014),
    203:("Outer Race",0.014), 204:("Outer Race",0.014),
    # Inner Race 0.021"
    213:("Inner Race",0.021), 214:("Inner Race",0.021),
    215:("Inner Race",0.021), 217:("Inner Race",0.021),
    # Ball 0.021"
    226:("Ball",0.021), 227:("Ball",0.021),
    228:("Ball",0.021), 229:("Ball",0.021),
    # Outer Race 0.021"
    238:("Outer Race",0.021), 239:("Outer Race",0.021),
    240:("Outer Race",0.021), 241:("Outer Race",0.021),
}

label_map   = {'Normal': 0, 'Inner Race': 1, 'Ball': 2, 'Outer Race': 3}
label_names = ['Normal', 'Inner Race', 'Ball', 'Outer Race']
# cnt = Counter(v[0] for v in file_map.values())
# for cls, n in sorted(cnt.items()):
#     print(f'{cls:15s}{n} files')


# ## CWRU MATLAB Files

# In[8]:


def load_recording(fp: Path) -> dict | None:
    num = int(fp.stem)
    if num not in file_map:
        return None
    label, defect_size = file_map[num]
    mat = loadmat(fp)

    de_keys = [k for k in mat if k.endswith('_DE_time')]
    if not de_keys:
        return None
    sig = np.asarray(mat[de_keys[0]]).squeeze().astype(np.float32)

    rpm_keys = [k for k in mat if k.endswith('RPM')]
    rpm_real = float(np.asarray(mat[rpm_keys[0]]).squeeze()) if rpm_keys else None


    default_rpm_by_file = {
        97: 1797.0,
        98: 1772.0,
        99: 1750.0,
        100: 1730.0,
    }
    if rpm_real is None:
        rpm_real = default_rpm_by_file.get(num, 1797.0)
        print(f'RPM missing in {fp.name}, using default {rpm_real:.1f}')

    return {
        'file_num':    num,
        'label':       label,
        'defect_size': defect_size,
        'signal':      sig,
        'rpm':         rpm_real,
        'fr':          rpm_real / 60.0,
        'n_samples':   len(sig),
    }

records = []
for fp in sorted(data_dir.glob('*.mat')):
    rec = load_recording(fp)
    if rec:
        records.append(rec)

print(f'{len(records)} CWRU recordings loaded')

df_meta = pd.DataFrame([{k: v for k, v in r.items() if k != 'signal'} for r in records])
print(f'RPM min = {df_meta.rpm.min():.2f}   RPM max = {df_meta.rpm.max():.2f}')
print(df_meta.label.value_counts().to_string())


# In[9]:


labels = list(set(r['label'] for r in records))
labels_s = random.sample(labels, 4)
print(labels_s)


# In[10]:


recs = []

for lab in labels_s:
    cand  = [r for r in records if r['label'] == lab]
    rec   = random.choice(cand)
    recs.append(rec)


fs = 48000

plt.figure(figsize=(12, 8))

for i, rec in enumerate(recs, 1):
    signal = rec['signal']


    start = np.random.randint(0, len(signal) - 2000)
    segment = signal[start:start+2000]

    t = np.arange(len(segment)) / fs

    plt.subplot(2, 2, i)
    plt.plot(t, segment)
    plt.title(f"{rec['label']} | RPM={rec['rpm']:.0f}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amp")
    plt.grid(True)

plt.tight_layout()
plt.show()


# ## Segmentation and Bearing Frequency Parameters

# In[11]:


FS          = 48_000
window_size = 16_384   # 2^14  -> Δf = 48000/16384 ≈ 2.93 Hz
window_step = 4_096    # overlap 75 %
band_width  = 10.0  

coef_BPFI = 5.4152
coef_BPFO = 3.5848
coef_BSF  = 4.7135
coef_FTF  = 0.39828

feature_names = [
    'rms', 'kurt', 'skewness', 'peak', 'crest_factor', 'std', 'mav',
    'dominant_freq', 'spec_centroid',
    'bpfo_energy', 'bpfi_energy', 'bsf_energy',
    'bpfo_h2_energy', 'bpfi_h2_energy', 'bsf_h2_energy',
    'bpfo_ratio', 'bpfi_ratio', 'bsf_ratio',
    'residual_energy', 'residual_ratio',
    'spec_std'
]
# Bearing fault-frequency coefficients (multipliers of shaft frequency fr)

BEARING_COEFFS = {
    "CWRU": {
        "BPFI": 5.4152, "BPFO": 3.5848, "BSF": 4.7135, "FTF": 0.39828
    },
    "Paderborn": {
        # FAG 6203: Nb=8, Bd=6.75mm, Pd=28.55mm, contact angle=0°
        "BPFI": 4.9561, "BPFO": 3.0439, "BSF": 1.9940, "FTF": 0.3805
    }
}

print(f' FFT Resolution : {FS/window_size:.3f} Hz')
print(f'Window duraton  : {window_size/FS*1000:.1f} ms')
fr_ex = 1796 / 60
print(f'BPFI at 1796 RPM : {coef_BPFI*fr_ex:.2f} Hz')
print(f'BPFO at 1796 RPM : {coef_BPFO*fr_ex:.2f} Hz')
print(f'BSF  at 1796 RPM : {coef_BSF *fr_ex:.2f} Hz')


# ## Signal Segmentation and Feature Extraction

# In[12]:


def segment(signal: np.ndarray, L: int = window_size, S: int = window_step) -> np.ndarray:
    n_windows = (len(signal) - L) // S + 1
    if n_windows <= 0:
        return np.array([])
    shape   = (n_windows, L)
    strides = (signal.strides[0] * S, signal.strides[0])
    return np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)


def fault_frequencies(fr: float, dataset: str = "CWRU") -> dict:
    c = BEARING_COEFFS[dataset]
    return {
        'f_BPFI': c["BPFI"] * fr,
        'f_BPFO': c["BPFO"] * fr,
        'f_BSF':  c["BSF"]  * fr,
    }

def lowpass_filter(signal, fs, cutoff=2000, order=4):
    sos = butter(order, cutoff, btype='low', fs=fs, output='sos')
    return sosfilt(sos, signal)

def extract_features(x: np.ndarray, fr: float, fs: int = FS,dataset: str = "CWRU") -> np.ndarray:

    if dataset == 'Paderborn':
      x = lowpass_filter(x, fs=fs, cutoff=2000)
    if dataset == 'Paderborn':
      band_width = 20.0
    else:
      band_width = 10.0

    x = x - np.mean(x)

    # Time-Domain
    rms          = np.sqrt(np.mean(x ** 2))
    kurt         = kurtosis(x, fisher=False)
    skewness     = skew(x)
    peak         = np.max(np.abs(x))
    crest_factor = peak / (rms + 1e-9)
    std          = np.std(x, ddof=1)
    mav          = np.mean(np.abs(x))

    # Frequency-domain
    window = np.hanning(len(x))
    gain   = np.sum(window)
    xw     = x * window
    N      = len(xw)
    X      = np.abs(rfft(xw)) / gain
    freqs  = rfftfreq(N, d=1.0 / fs)

    dominant_freq = freqs[np.argmax(X)]
    spec_centroid = np.sum(freqs * X) / (np.sum(X) + 1e-9)
    spec_std      = float(np.std(X, ddof=1))

    ff = fault_frequencies(fr, dataset = dataset)

    def band_energy(f0: float, df: float = band_width) -> float:
        mask = np.abs(freqs - f0) <= df
        return float(np.sum(X[mask] ** 2))

    bpfo_energy = band_energy(ff['f_BPFO'])
    bpfi_energy = band_energy(ff['f_BPFI'])
    bsf_energy  = band_energy(ff['f_BSF'])

    # Harmonics
    bpfo_h2_energy = band_energy(2 * ff['f_BPFO'])
    bpfi_h2_energy = band_energy(2 * ff['f_BPFI'])
    bsf_h2_energy  = band_energy(2 * ff['f_BSF'])

    # Ratios
    total_spec_energy = float(np.sum(X**2) + 1e-9)

    bpfo_ratio = bpfo_energy / total_spec_energy
    bpfi_ratio = bpfi_energy / total_spec_energy
    bsf_ratio  = bsf_energy  / total_spec_energy

    fault_energy    = bpfo_energy + bpfi_energy + bsf_energy
    residual_energy = max(total_spec_energy - fault_energy, 0.0)
    residual_ratio  = residual_energy / total_spec_energy

    return np.array([
        rms, kurt, skewness, peak, crest_factor, std, mav,
        dominant_freq, spec_centroid,
        bpfo_energy, bpfi_energy, bsf_energy,
        bpfo_h2_energy, bpfi_h2_energy, bsf_h2_energy,
        bpfo_ratio, bpfi_ratio, bsf_ratio,
        residual_energy, residual_ratio,
        spec_std,
    ], dtype=np.float32)


# ## 9. CWRU Feature Matrix

# In[13]:


rows = []
for rec in tqdm(records, desc='Feature extraction'):
    windows = segment(rec['signal'])
    for window in windows:
        features = extract_features(window, fr=rec['fr'])
        row = {name: val for name, val in zip(feature_names, features)}
        row.update({
            'label':       rec['label'],
            'defect_size': rec['defect_size'],
            'rpm':         rec['rpm'],
            'fr':          rec['fr'],
            'file_num':    rec['file_num'],
        })
        rows.append(row)

df = pd.DataFrame(rows)
df['y'] = df['label'].map(label_map)
print(df.label.value_counts().to_string())
df.to_parquet(output_dir / 'features.parquet', index=False)
df.head()


# ## Exploratory Data Analysis

# In[15]:


colors = {'Normal':'green', 'Inner Race':'blue', 'Ball':'orange', 'Outer Race':'red'}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

counts = df['label'].value_counts()
axes[0].bar(counts.index, counts.values, color=[colors[l] for l in counts.index])
axes[0].set_title('Class Distribution')
axes[0].set_ylabel('Number of windows')
axes[0].grid()

for label, grp in df.groupby('label'):
    axes[1].hist(grp['kurt'].clip(-5, 30), bins=60,
                 alpha=0.55, label=label, color=colors[label])
axes[1].set_title('Kurtosis distribution by class')
axes[1].set_xlabel('Kurtosis')
axes[1].legend()
axes[1].grid()

for label, grp in df.groupby('label'):
    axes[2].hist(np.log1p(grp['bpfi_energy']), bins=60,
                 alpha=0.55, label=label, color=colors[label])
axes[2].set_title('log(1 + BPFI energy) distribution by class')
axes[2].set_xlabel('log(1 + BPFI energy)')
axes[2].legend()
axes[2].grid()

plt.tight_layout()
plt.savefig(fig_dir / 'eda.png', dpi=150)

plt.show()


# In[16]:


fig, ax = plt.subplots(figsize=(10, 8))
corr = df[feature_names].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            ax=ax, linewidths=0.4)
ax.set_title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig(fig_dir / 'feature_corr.png', dpi=150)
plt.show()


# ## File-Level Train / Validation / Test Split

# In[17]:


from sklearn.model_selection import train_test_split

train_files = []
val_files = []
test_files = []

for label in df['label'].unique():
    files = np.array(sorted(df.loc[df['label'] == label, 'file_num'].unique()))

    f_train_val, f_test = train_test_split(
        files, test_size=0.20, random_state=42
    )
    f_train, f_val = train_test_split(
        f_train_val, test_size=0.10, random_state=42
    )

    train_files.extend(f_train.tolist())
    val_files.extend(f_val.tolist())
    test_files.extend(f_test.tolist())

# Window-level dataframes inherit the file-level split.
df_train = df[df['file_num'].isin(train_files)].reset_index(drop=True)
df_val   = df[df['file_num'].isin(val_files)].reset_index(drop=True)
df_test  = df[df['file_num'].isin(test_files)].reset_index(drop=True)

n_files_total = len(set(train_files) | set(val_files) | set(test_files))
print('Window counts:')
print('  Train:', len(df_train))
print('  Val  :', len(df_val))
print('  Test :', len(df_test))
print('\nRecording-file counts and realized proportions:')
for name, files_ in [('Train', train_files), ('Validation', val_files), ('Test', test_files)]:
    n = len(set(files_))
    print(f'  {name:10s}: {n:2d}/{n_files_total} = {100*n/n_files_total:.1f}%')

split_table = pd.DataFrame({
    'split': ['Train', 'Validation', 'Test'],
    'n_files': [len(set(train_files)), len(set(val_files)), len(set(test_files))],
    'n_windows': [len(df_train), len(df_val), len(df_test)],
})
split_table['file_percent'] = 100 * split_table['n_files'] / n_files_total
split_table.to_csv(output_dir / 'cwru_file_level_split.csv', index=False)
print(split_table)


# In[18]:


train_files = set(df_train['file_num'].unique())
val_files   = set(df_val['file_num'].unique())
test_files  = set(df_test['file_num'].unique())

print("Train ∩ Val  =", len(train_files & val_files))
print("Train ∩ Test =", len(train_files & test_files))
print("Val ∩ Test   =", len(val_files & test_files))


# In[19]:


print("\nDistribution train:")
print(df_train['label'].value_counts())

print("\nDistribution val:")
print(df_val['label'].value_counts())

print("\nDistribution test:")
print(df_test['label'].value_counts())


# ## Sensor-to-Text Encoding and Diagnostic Template

# In[20]:


def cwru_load_from_rpm(rpm: float) -> str:
    """Map measured/nominal CWRU speed to motor load."""
    if 1790 <= rpm <= 1805: return '0 HP'
    if 1765 <= rpm <= 1785: return '1 HP'
    if 1740 <= rpm <= 1760: return '2 HP'
    if 1715 <= rpm <= 1740: return '3 HP'
    return 'Unknown'


def encode_to_text_variant(row,
                           include_operating_metadata: bool = True,
                           include_expected_fault_freqs: bool = True) -> str:
    """Sensor-to-text encoder used for the revision experiments.

    Full representation contains the 21 numerical features PLUS operating
    metadata (RPM/load/shaft frequency) and the physically expected fault
    frequencies. Turning either component off supports controlled ablations.
    """
    lines = [
        'Vibration features:',
        f"- RMS: {row['rms']:.4f}",
        f"- Kurtosis: {row['kurt']:.4f}",
        f"- Skewness: {row['skewness']:.4f}",
        f"- Peak: {row['peak']:.4f}",
        f"- Crest factor: {row['crest_factor']:.4f}",
        f"- Std: {row['std']:.4f}",
        f"- MAV: {row['mav']:.4f}",
        f"- Dominant frequency: {row['dominant_freq']:.4f} Hz",
        f"- Spectral centroid: {row['spec_centroid']:.4f} Hz",
        f"- BPFO energy: {row['bpfo_energy']:.6e}",
        f"- BPFI energy: {row['bpfi_energy']:.6e}",
        f"- BSF energy: {row['bsf_energy']:.6e}",
        f"- BPFO h2 energy: {row['bpfo_h2_energy']:.6e}",
        f"- BPFI h2 energy: {row['bpfi_h2_energy']:.6e}",
        f"- BSF h2 energy: {row['bsf_h2_energy']:.6e}",
        f"- BPFO ratio: {row['bpfo_ratio']:.6e}",
        f"- BPFI ratio: {row['bpfi_ratio']:.6e}",
        f"- BSF ratio: {row['bsf_ratio']:.6e}",
        f"- Residual energy: {row['residual_energy']:.6e}",
        f"- Residual ratio: {row['residual_ratio']:.6e}",
        f"- Spectral std: {row['spec_std']:.6e}",
    ]

    if include_operating_metadata:
        load = row['load'] if 'load' in row and pd.notna(row['load']) else cwru_load_from_rpm(float(row['rpm']))
        lines += [
            'Operating metadata:',
            f"- Motor load: {load}",
            f"- Shaft speed: {float(row['rpm']):.1f} RPM",
            f"- Shaft frequency: {float(row['fr']):.4f} Hz",
        ]

    if include_expected_fault_freqs:
        ff = fault_frequencies(float(row['fr']), dataset='CWRU')
        lines += [
            'Expected bearing fault frequencies at this operating point:',
            f"- Expected BPFO: {ff['f_BPFO']:.2f} Hz",
            f"- Expected BPFI: {ff['f_BPFI']:.2f} Hz",
            f"- Expected BSF: {ff['f_BSF']:.2f} Hz",
        ]

    return '\n'.join(lines) + '\n'


def encode_to_text(row):
    """Full proposed representation used by the main model."""
    return encode_to_text_variant(
        row,
        include_operating_metadata=True,
        include_expected_fault_freqs=True,
    )


def make_output(row) -> str:
    """Deterministic target-template generator (NOT another LLM)."""
    label = row['label']
    size  = row['defect_size']
    ff    = fault_frequencies(row['fr'])

    if label == 'Normal':
        return (
            f"Diagnosis: Normal bearing. Severity: None. "
            f"Explanation: Kurtosis={row['kurt']:.2f} is close to 3 (Gaussian baseline), "
            f"and all fault-band energies (BPFO, BPFI, BSF) are low, "
            f"indicating no impulsive fault signatures. "
            f"Recommendation: Continue routine monitoring schedule. Urgency: None."
        )

    sev = 'Low' if size <= 0.007 else ('Medium' if size <= 0.014 else 'High')

    fault_details = {
        'Inner Race': (
            f"Elevated BPFI band energy ({row['bpfi_energy']:.3e}) "
            f"at expected frequency {ff['f_BPFI']:.1f} Hz",
            'Inspect inner raceway surface; plan bearing replacement'
        ),
        'Outer Race': (
            f"Elevated BPFO band energy ({row['bpfo_energy']:.3e}) "
            f"at expected frequency {ff['f_BPFO']:.1f} Hz",
            'Inspect outer raceway; plan replacement at next maintenance window'
        ),
        'Ball': (
            f"Elevated BSF band energy ({row['bsf_energy']:.3e}) "
            f"at expected frequency {ff['f_BSF']:.1f} Hz with rising crest factor",
            'Inspect rolling elements; plan bearing replacement'
        ),
    }
    why, action = fault_details[label]
    urg_map = {
        'Low':    'Low - schedule inspection within 1 month',
        'Medium': 'Medium - schedule replacement within 2 weeks',
        'High':   'High - replace immediately to avoid unplanned downtime',
    }

    return (
        f"Diagnosis: {label} fault. Severity: {sev}. "
        f"Explanation: {why} and kurtosis={row['kurt']:.2f} suggest "
        f"impulsive defect on the {label.lower()}. "
        f"Recommendation: {action}. Urgency: {urg_map[sev]}."
    )

# Add metadata once so all downstream datasets use the same definition.
for d in [df_train, df_val, df_test]:
    d['load'] = d['rpm'].apply(cwru_load_from_rpm)
    d['text'] = d.apply(encode_to_text, axis=1)
    d['diagnosis'] = d.apply(make_output, axis=1)

print(df_train[['label', 'load', 'text', 'diagnosis']].head(2))


# ## Retrieval-Augmented Generation Index with FAISS

# In[21]:


rag_corpus = df_train["text"].tolist()
rag_labels = df_train["label"].tolist()
rag_file_nums = df_train["file_num"].tolist()

print("RAG corpus size:", len(rag_corpus))
print("Unique train files in RAG:", len(set(rag_file_nums)))


# ## Sentence Embedding Model

# In[22]:


from huggingface_hub import login

# notebook_login()
login("")

embedder = SentenceTransformer("all-MiniLM-L6-v2", device=("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"))

train_embeddings = embedder.encode(
    rag_corpus,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
).astype("float32")

print(train_embeddings.shape)


# In[23]:


dim = train_embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(train_embeddings)

print("FAISS index size:", index.ntotal)


# In[24]:


def retrieve(query_text: str, k: int = 3) -> list[dict]:
    """Retrieve the k most similar training cases."""
    q = embedder.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(q, k)

    results = []
    for s, i in zip(scores[0], ids[0]):
        i = int(i)
        results.append({
            "text": df_train.iloc[i]["text"],
            "label": df_train.iloc[i]["label"],
            "diagnosis": df_train.iloc[i]["diagnosis"],
            "file_num": df_train.iloc[i]["file_num"],
            "score": float(s)
        })
    return results


# In[25]:


sample = df_test.iloc[0]
retrieved = retrieve(sample["text"], k=3)

print("True label:", sample["label"], "| file:", sample["file_num"])
for r in retrieved:
    print(
        f"score={r['score']:.4f} | "
        f"label={r['label']} | "
        f"file={r['file_num']}"
    )


# In[26]:


def retrieval_diagnostics(df_subset, k_values=(1, 3), retrieve_fn=retrieve):
    """Evaluate retrieval on the COMPLETE subset, not an arbitrary 100-window sample."""
    rows = []
    for _, row in tqdm(df_subset.iterrows(), total=len(df_subset), desc='Retrieval diagnostics'):
        kmax = max(k_values)
        retrieved_all = retrieve_fn(row['text'], k=kmax + 8)
        retrieved = [r for r in retrieved_all if r['file_num'] != row['file_num']][:kmax]
        rec = {
            'query_file': row['file_num'],
            'true_label': row['label'],
            'query_load': row.get('load', cwru_load_from_rpm(row['rpm'])),
        }
        for k in k_values:
            topk = retrieved[:k]
            rec[f'hit@{k}'] = int(any(r['label'] == row['label'] for r in topk))
            rec[f'top{k}_labels'] = '|'.join(r['label'] for r in topk)
        rows.append(rec)
    details = pd.DataFrame(rows)
    summary = {f'Hit@{k}': details[f'hit@{k}'].mean() for k in k_values}
    return details, summary

retrieval_test_details, retrieval_test_summary = retrieval_diagnostics(df_test, k_values=(1, 3))
print(retrieval_test_summary)
retrieval_test_details.to_csv(output_dir / 'retrieval_diagnostics_cwru_test.csv', index=False)


# ## Instruction Dataset Construction

# In[27]:


SYSTEM_PROMPT = (
    "You are an expert industrial maintenance assistant specialized in rolling-element "
    "bearing fault diagnosis in the context of Industry 4.0 predictive maintenance. "
    "Given vibration-derived features from a CWRU bearing sensor and similar retrieved "
    "historical cases, produce a structured diagnostic report with exactly four components: "
    "fault class, severity, explanation, and maintenance recommendation with urgency level."
)


def build_prompt(query_text: str, retrieved: list = None) -> str:

    ctx = ""
    if retrieved:
        ctx_lines = []
        for i, r in enumerate(retrieved, 1):
            ctx_lines.append(
                f"[Retrieved case {i}] (similarity={r['score']:.3f})\n"
                f"Features: {r['text']}\n"
                f"Diagnosis: {r['diagnosis']}"
            )
        ctx = "### Similar retrieved historical cases:\n" + "\n\n".join(ctx_lines) + "\n\n"

    return (
        f"{ctx}"
        f"### Current sensor reading:\n{query_text}\n"
        f"Return exactly this format:\n"
        f"Fault class: [Normal | Ball | Inner Race | Outer Race]\n"
        f"Severity: [None | Low | Medium | High]\n"
        f"Explanation: [which signal features justify the diagnosis]\n"
        f"Recommendation: [maintenance action and urgency level]\n"
    )


def make_dataset(split_df: pd.DataFrame,
                 use_rag: bool = True,
                 n_max: int = None) -> list:
    """Build instruction-tuning samples, with or without RAG."""
    if n_max is not None:
        split_df = split_df.sample(min(n_max, len(split_df)), random_state=42)

    samples = []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc='Building dataset'):
        diag  = row['diagnosis']
        label = row['label']

        sev_m = re.search(r'Severity:\s*(\w+)', diag)
        severity = sev_m.group(1) if sev_m else 'None'

        exp_m = re.search(r'Explanation:\s*(.+?)(?:\s+Recommendation:)', diag, re.DOTALL)
        explanation = exp_m.group(1).strip() if exp_m else diag

        rec_m = re.search(r'Recommendation:\s*(.+?)(?:\s+Urgency:)', diag, re.DOTALL)
        urg_m = re.search(r'Urgency:\s*(.+)', diag)
        if rec_m and urg_m:
            recommendation = f"{rec_m.group(1).strip()}. Urgency: {urg_m.group(1).strip()}"
        elif rec_m:
            recommendation = rec_m.group(1).strip()
        else:
            recommendation = "Continue routine monitoring."

        completion = (
            f"Fault class: {label}\n"
            f"Severity: {severity}\n"
            f"Explanation: {explanation}\n"
            f"Recommendation: {recommendation}"
        )


        retrieved = None
        if use_rag:
            retrieved_all = retrieve(row['text'], k=4)
            retrieved = [r for r in retrieved_all
                         if r['file_num'] != row['file_num']][:3]

        samples.append({
            'prompt':     build_prompt(row['text'], retrieved),
            'completion': completion,
            'label':      label,
        })
    return samples


n_train_llm = len(df_train)
n_val_llm   = len(df_val)
n_test_llm  = len(df_test)

train_lm = make_dataset(df_train, use_rag=True, n_max=n_train_llm)
val_lm   = make_dataset(df_val,   use_rag=True, n_max=n_val_llm)
test_lm  = make_dataset(df_test,  use_rag=True, n_max=n_test_llm)

print(f"Train samples : {len(train_lm)}")
print(f"Val samples   : {len(val_lm)}")
print(f"Test samples  : {len(test_lm)}")

print('\nExample of prompt with RAG')
print(train_lm[0]['prompt'][:1500], '...')

print('\nExpected Completion ')
print(train_lm[0]['completion'])


# ## LLaMA 3.2 3B LoRA Fine-Tuning in 4-bit
# 

# In[28]:


try:
    del embedder
except NameError:
    pass
gc.collect()
torch.cuda.empty_cache()

base_model_id = 'meta-llama/Llama-3.2-3B-Instruct'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.float16
)
# In[29]:


tokenizer = AutoTokenizer.from_pretrained(base_model_id,clean_up_tokenization_spaces=False)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = 'right'

# In[30]:

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
)
# In[31]:

MAX_SEQ_LEN = 1536

RESPONSE_TEMPLATE = "<|start_header_id|>assistant<|end_header_id|>\n\n"

def format_for_training(sample: dict) -> dict:
    msgs = [
        {'role': 'system',    'content': SYSTEM_PROMPT},
        {'role': 'user',      'content': sample['prompt']},
        {'role': 'assistant', 'content': sample['completion']},
    ]
    text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False
    )
    return {'text': text}


train_ds = Dataset.from_list(train_lm).map(
    format_for_training,
    remove_columns=['prompt', 'completion', 'label'],
)

val_ds = Dataset.from_list(val_lm).map(
    format_for_training,
    remove_columns=['prompt', 'completion', 'label'],
)

print(f"Train dataset formated: {len(train_ds)} samples")
print(f"Val dataset formated   : {len(val_ds)} samples")
print(train_ds[0]['text'][:600], '...')

# In[32]:

from transformers import DataCollatorForLanguageModeling, set_seed

ASSISTANT_HEADER_IDS = tokenizer.encode(
    '<|start_header_id|>assistant<|end_header_id|>\n\n',
    add_special_tokens=False
)

class CompletionOnlyCollator(DataCollatorForLanguageModeling):
    """Mask all prompt tokens; compute loss only on the assistant completion."""
    def __init__(self, tokenizer, response_ids):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.response_ids = list(response_ids)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        labels = batch['labels'].clone()
        tpl = self.response_ids
        n_tpl = len(tpl)
        for i in range(labels.size(0)):
            seq = batch['input_ids'][i].tolist()
            start = None
            for j in range(len(seq) - n_tpl + 1):
                if seq[j:j+n_tpl] == tpl:
                    start = j + n_tpl
                    break
            if start is None:
                labels[i, :] = -100
            else:
                labels[i, :start] = -100
        batch['labels'] = labels
        return batch

collator = CompletionOnlyCollator(tokenizer, ASSISTANT_HEADER_IDS)


def train_lora_adapter(train_dataset, val_dataset, out_dir, seed=42, epochs=5):
    """Train one independent QLoRA adapter from the same base model."""
    set_seed(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    gc.collect(); torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map='auto',
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation='eager',
    )
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    model = get_peft_model(model, lora_cfg)

    args = SFTConfig(
        output_dir=str(out_dir) + '_checkpoints',
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_accumulation_steps=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=50,
        lr_scheduler_type='cosine',
        logging_steps=10,
        logging_strategy='steps',
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False,
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
        optim='paged_adamw_8bit',
        report_to='none',
        dataset_text_field='text',
        max_length=MAX_SEQ_LEN,
        packing=False,
        seed=seed,
        data_seed=seed,
        max_grad_norm=1.0,
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=args,
        data_collator=collator,
    )
    print(f'Training seed={seed} -> {out_dir}')
    train_output = trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    metrics = dict(train_output.metrics)
    metrics['seed'] = seed
    with open(str(out_dir) + '_training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    del trainer, model
    gc.collect(); torch.cuda.empty_cache()
    return metrics


# MAIN_ADAPTER_DIR = output_dir / 'llama32_lora_revision_seed42'
# train_lora_adapter(train_ds, val_ds, MAIN_ADAPTER_DIR, seed=42, epochs=5)

# ## LLM Inference on CWRU

# In[33]:

# gc.collect()
# torch.cuda.empty_cache()

# lora_path = str(MAIN_ADAPTER_DIR)
# base_obj = AutoModelForCausalLM.from_pretrained(
#     base_model_id,
#     device_map={'': 0},
#     quantization_config=bnb_config,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
#     attn_implementation='eager',
# )
# model_inf = PeftModel.from_pretrained(base_obj, lora_path)
# model_inf.eval()

# tokenizer_inf = AutoTokenizer.from_pretrained(lora_path)
# tokenizer_inf.pad_token = tokenizer_inf.eos_token
# tokenizer_inf.padding_side = 'right'

# In[34]:
def llm_predict(prompt_text: str, max_new_tokens: int = 150) -> str:
    msgs = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user',   'content': prompt_text},
    ]
    enc = tokenizer_inf.apply_chat_template(
        msgs,
        add_generation_prompt=True,
        return_tensors='pt',
        return_dict=True
    ).to(model_inf.device)

    prompt_len = enc["input_ids"].shape[1]

    with torch.inference_mode():
        out = model_inf.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer_inf.eos_token_id,
            eos_token_id=tokenizer_inf.eos_token_id,
            use_cache=True,
            repetition_penalty=1.1,
        )

    gen_tokens = out[0][prompt_len:]
    decoded = tokenizer_inf.decode(gen_tokens, skip_special_tokens=True).strip()
    del enc, out, gen_tokens
    torch.cuda.empty_cache()
    return decoded
# In[35]:

def parse_class(text: str) -> str:
    patterns = [
        r"Fault class:\s*<?(['\"]?)(Normal|Ball|Inner Race|Outer Race)",
        r"####\s*Fault class:\s*\n\s*(Normal|Ball|Inner Race|Outer Race)",
        r"Fault class[:\s]*\n\s*(Normal|Ball|Inner Race|Outer Race)",
        r"\b(Normal|Ball|Inner Race|Outer Race)\b"
    ]
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            pred = m.groups()[-1].strip().strip('<>').strip()
            mapping = {
                'normal':     'Normal',
                'ball':       'Ball',
                'inner race': 'Inner Race',
                'outer race': 'Outer Race',
            }
            return mapping.get(pred.lower(), 'Unknown')
    return 'Unknown'

def parse_severity(text: str) -> str:
    m = re.search(r"(?:####\s*Severity:\s*\n\s*|Severity:\s*<?)(None|Low|Medium|High)", text, flags=re.IGNORECASE)
    return m.group(1).capitalize() if m else 'Unknown'

def parse_explanation(text: str) -> str:
    """Extract explanation from the 4-component generated output."""
    m = re.search(r"Explanation:\s*(.+?)(?:\nRecommendation:|$)", text, re.DOTALL)
    return m.group(1).strip() if m else ''

def parse_recommendation(text: str) -> str:
    """Extract recommendation from the 4-component generated output."""
    m = re.search(r"Recommendation:\s*(.+?)$", text, re.DOTALL)
    return m.group(1).strip() if m else ''


# In[36]:


# model_results = []
# for i, sample in enumerate(tqdm(test_lm, desc='Revised CWRU LLM inference')):
#     gen = llm_predict(sample['prompt'])
#     model_results.append({
#         'true_label': sample['label'],
#         'pred_label': parse_class(gen),
#         'pred_severity': parse_severity(gen),
#         'pred_explanation': parse_explanation(gen),
#         'pred_recommendation': parse_recommendation(gen),
#         'generated': gen,
#     })
#     if (i + 1) % 50 == 0:
#         with open(output_dir / 'model_results_revision_partial.json', 'w') as f:
#             json.dump(model_results, f, ensure_ascii=False, indent=2)

# with open(output_dir / 'model_results_revision.json', 'w', encoding='utf-8') as f:
#     json.dump(model_results, f, ensure_ascii=False, indent=2)
# print(f'{len(model_results)} revised CWRU diagnostics generated.')


# In[37]:


with open(output_dir / 'model_results_revision.json', 'r', encoding='utf-8') as f:
    model_results = json.load(f)
res_df = pd.DataFrame(model_results)
valid_mask = res_df['pred_label'] != 'Unknown'
res_eval = res_df[valid_mask].copy()
print(f"Parsed predictions: {len(res_eval)}/{len(res_df)}")

acc = accuracy_score(res_eval['true_label'], res_eval['pred_label'])
prec = precision_score(res_eval['true_label'], res_eval['pred_label'], average='macro', zero_division=0)
rec = recall_score(res_eval['true_label'], res_eval['pred_label'], average='macro', zero_division=0)
f1 = f1_score(res_eval['true_label'], res_eval['pred_label'], average='macro', zero_division=0)
print(f"LLM Accuracy : {acc*100:.2f}%")
print(f"LLM Precision: {prec*100:.2f}%")
print(f"LLM Recall   : {rec*100:.2f}%")
print(f"LLM F1       : {f1*100:.2f}%")
print('\nClass-specific report:\n')
print(classification_report(res_eval['true_label'], res_eval['pred_label'], zero_division=0))

pd.DataFrame(classification_report(
    res_eval['true_label'], res_eval['pred_label'], output_dict=True, zero_division=0
)).T.to_csv(output_dir / 'cwru_llm_classification_report_revision.csv')


# ## Classical Baselines: SVM and Random Forest

# In[38]:

feature_cols = feature_names

X_train = df_train[feature_cols].values
X_val   = df_val[feature_cols].values
X_test  = df_test[feature_cols].values

y_train = df_train['y'].values
y_val   = df_val['y'].values
y_test  = df_test['y'].values

scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

print(f'X_train_sc : {X_train_sc.shape}')
print(f'X_val_sc   : {X_val_sc.shape}')
print(f'X_test_sc  : {X_test_sc.shape}')


# ## SVM and Random Forest Evaluation

# In[39]:


def evaluate(clf, X_test, y_test, name: str) -> dict:
    y_pred    = clf.predict(X_test)
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall    = recall_score(y_test,   y_pred, average='macro', zero_division=0)
    f1        = f1_score(y_test,       y_pred, average='macro', zero_division=0)
    print(f'\n{name} Results')
    print(f'  Accuracy  : {accuracy*100:.2f}%')
    print(f'  Precision : {precision*100:.2f}%')
    print(f'  Recall    : {recall*100:.2f}%')
    print(f'  F1 Score  : {f1*100:.2f}%')
    return {
        'model':     name,
        'accuracy':  round(accuracy  * 100, 2),
        'precision': round(precision * 100, 2),
        'recall':    round(recall    * 100, 2),
        'f1_score':  round(f1        * 100, 2),
        'y_true':    y_test.tolist(),
        'y_pred':    y_pred.tolist(),
    }

def plot_cm(y_true, y_pred, name: str, save_path: Path):
    cm     = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names,
                linewidths=0.4, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{name}  Confusion matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


# #### Support Vector Machine

# In[40]:

df_dev = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

X_dev = df_dev[feature_names].values
y_dev = df_dev['y'].values
groups_dev = df_dev['file_num'].values


from sklearn.preprocessing import StandardScaler

gkf = GroupKFold(n_splits=4)

param_grid = {
    'C': [0.1, 1, 10, 100,1000],
    'gamma': ['scale', 0.01, 0.001,0.0001]
}

best_score = -1
best_params = None

for params in ParameterGrid(param_grid):
    fold_scores = []

    for train_idx, val_idx in gkf.split(X_dev, y_dev, groups=groups_dev):
        X_tr, X_va = X_dev[train_idx], X_dev[val_idx]
        y_tr, y_va = y_dev[train_idx], y_dev[val_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)

        model = SVC(kernel='rbf', C=params['C'], gamma=params['gamma'], random_state=42)
        model.fit(X_tr_sc, y_tr)

        y_pred = model.predict(X_va_sc)
        fold_scores.append(accuracy_score(y_va, y_pred))

    mean_score = np.mean(fold_scores)
    print(f"SVM {params} -> CV acc = {mean_score*100:.2f}%")

    if mean_score > best_score:
        best_score = mean_score
        best_paramssvm = params

print("\nBest SVM params:", best_paramssvm)
print(f"Best CV accuracy: {best_score*100:.2f}%")


# In[41]:


scaler = StandardScaler()
X_trainval_sc = scaler.fit_transform(X_dev)
X_test_sc = scaler.transform(X_test)

svm = SVC(kernel='rbf', C=best_paramssvm['C'], gamma=best_paramssvm['gamma'], random_state=42)
svm.fit(X_trainval_sc, y_dev)

svm_results = evaluate(
    svm,
    X_test_sc,
    y_test,
    f"SVM GroupCV (C={best_paramssvm['C']}, gamma={best_paramssvm['gamma']})"
)

plot_cm(svm_results['y_true'], svm_results['y_pred'], 'SVM GroupCV', fig_dir / 'svm_groupcv_cm.png')

# #### Random Forest

# In[42]:
df_dev = pd.concat([df_train, df_val], axis=0).reset_index(drop=True)

X_dev = df_dev[feature_names].values
y_dev = df_dev['y'].values
groups_dev = df_dev['file_num'].values

gkf = GroupKFold(n_splits=4)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20]
}

best_score = -1
best_params = None

for params in ParameterGrid(param_grid):
    fold_scores = []

    for train_idx, val_idx in gkf.split(X_dev, y_dev, groups=groups_dev):
        X_tr, X_va = X_dev[train_idx], X_dev[val_idx]
        y_tr, y_va = y_dev[train_idx], y_dev[val_idx]

        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        fold_scores.append(accuracy_score(y_va, y_pred))

    mean_score = np.mean(fold_scores)
    print(f"RF {params} -> CV acc = {mean_score*100:.2f}%")

    if mean_score > best_score:
        best_score = mean_score
        best_paramsrf = params

print("\nBest RF params:", best_paramsrf)
print(f"Best CV accuracy: {best_score*100:.2f}%")


# In[43]:


rf = RandomForestClassifier(
    n_estimators=best_paramsrf['n_estimators'],
    max_depth=best_paramsrf['max_depth'],
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf.fit(X_dev, y_dev)

rf_results = evaluate(
    rf,
    X_test,
    y_test,
    f"RF GroupCV ({best_paramsrf['n_estimators']} trees, max_depth={best_paramsrf['max_depth']})"
)

plot_cm(rf_results['y_true'], rf_results['y_pred'], 'RF GroupCV', fig_dir / 'rf_groupcv_cm.png')


# ## Random Forest Feature Importance

# In[44]:


importances = pd.DataFrame({
    'feature':    feature_names,
    'importance': rf.feature_importances_,
}).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importances['feature'], importances['importance'], color='steelblue')
ax.set_title('Random Forest - Feature Importance')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(fig_dir / 'rf_feature_importance.png', dpi=150)
plt.show()

# ## Comparative Evaluation on CWRU


with open(output_dir / 'model_results_revision.json', 'r', encoding='utf-8') as f:
    model_results = json.load(f)
llm_true = np.array([label_map[r['true_label']] for r in model_results if r['pred_label'] != 'Unknown'])
llm_pred = np.array([label_map[r['pred_label']] for r in model_results if r['pred_label'] != 'Unknown'])
llm_results = {
    'model': 'LLaMA 3.2 3B + LoRA + RAG (revised encoding)',
    'accuracy': round(accuracy_score(llm_true, llm_pred) * 100, 2),
    'precision': round(precision_score(llm_true, llm_pred, average='macro', zero_division=0) * 100, 2),
    'recall': round(recall_score(llm_true, llm_pred, average='macro', zero_division=0) * 100, 2),
    'f1_score': round(f1_score(llm_true, llm_pred, average='macro', zero_division=0) * 100, 2),
}
print('LLM Results:', llm_results)
plot_cm(llm_true, llm_pred, 'LLaMA 3.2 LoRA revised', fig_dir / 'llm_cm_revision.png')

all_results = [svm_results, rf_results, llm_results]

df_results = pd.DataFrame([{
    'Model':     r['model'],
    'Accuracy':  r['accuracy'],
    'Precision': r['precision'],
    'Recall':    r['recall'],
    'F1 Score':  r['f1_score'],
} for r in all_results])


print(df_results.to_string(index=False))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
x       = np.arange(len(metrics))
width   = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
for i, row in df_results.iterrows():
    ax.bar(x + i * width, [row[m] for m in metrics], width, label=row['Model'])

ax.set_xticks(x + width)
ax.set_xticklabels(metrics)
ax.set_ylabel('Score (%)')
ax.set_title('Model Comparison - CWRU Bearing Fault Diagnosis')
ax.set_ylim(0, 110)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(fig_dir / 'model_comparison.png', dpi=150)
plt.show()

df_results.to_csv(output_dir / 'evaluation_results.csv', index=False)

import pickle
with open('best_svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)


# ## Readability Metrics for Generated Reports
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
import textstat

with open(output_dir / 'model_results_revision.json', 'r', encoding='utf-8') as f:
    model_results = json.load(f)

def extract_explanation_text(generated: str) -> str:
    m_exp = re.search(r'Explanation:\s*(.+?)(?=\nRecommendation:|$)', generated, re.DOTALL)
    m_rec = re.search(r'Recommendation:\s*(.+?)$', generated, re.DOTALL)
    parts = []
    if m_exp: parts.append(m_exp.group(1).strip())
    if m_rec: parts.append(m_rec.group(1).strip())
    return ' '.join(parts)

texts_by_class = {'Normal': [], 'Inner Race': [], 'Outer Race': [], 'Ball': []}
texts_by_sev   = {'None': [], 'Low': [], 'Medium': [], 'High': []}

for r in model_results:
    text = extract_explanation_text(r['generated'])
    if len(text.split()) < 5:
        continue
    cls = r.get('pred_label', 'Unknown')
    sev = r.get('pred_severity', 'Unknown')
    if cls in texts_by_class:
        texts_by_class[cls].append(text)
    if sev in texts_by_sev:
        texts_by_sev[sev].append(text)

def compute_readability(texts: dict, group_name: str) -> pd.DataFrame:
    rows = []
    for key, txts in texts.items():
        if not txts:
            continue
        agg = ' '.join(txts)
        if len(agg.split()) < 100:
            print(f"  {key}: moins de 100 mots ({len(agg.split())}), scores omis")
            continue
        rows.append({
            group_name:    key,
            'N_diag':      len(txts),
            'Words':       len(agg.split()),
            'Flesch RE':   round(textstat.flesch_reading_ease(agg), 2),
            'FK Grade':    round(textstat.flesch_kincaid_grade(agg), 2),
            'Gunning Fog': round(textstat.gunning_fog(agg), 2),
            'SMOG':        round(textstat.smog_index(agg), 2),
            'Dale-Chall':  round(textstat.dale_chall_readability_score(agg), 2),
        })
    return pd.DataFrame(rows)

print("Readability by CLASS")
df_read_cls = compute_readability(texts_by_class, 'Class')
print(df_read_cls.to_string(index=False))

print("\nReadability by SEVERITY")
df_read_sev = compute_readability(texts_by_sev, 'Severity')
print(df_read_sev.to_string(index=False))

all_texts_flat = [t for txts in texts_by_class.values() for t in txts]
all_text_joined = ' '.join(all_texts_flat)

if len(all_text_joined.split()) >= 100:
    fk_global  = textstat.flesch_kincaid_grade(all_text_joined)
    fre_global = textstat.flesch_reading_ease(all_text_joined)

    global_row = {
        'Class':       'ALL (global)',
        'N_diag':      len(model_results),
        'Words':       len(all_text_joined.split()),
        'Flesch RE':   round(fre_global, 2),
        'FK Grade':    round(fk_global, 2),
        'Gunning Fog': round(textstat.gunning_fog(all_text_joined), 2),
        'SMOG':        round(textstat.smog_index(all_text_joined), 2),
        'Dale-Chall':  round(textstat.dale_chall_readability_score(all_text_joined), 2),
    }

    df_read_full = pd.concat([df_read_cls, pd.DataFrame([global_row])], ignore_index=True)
    df_read_full.to_csv(output_dir / 'readability_scores_full.csv', index=False)

    print("\nGlobal Score")
    print(f"Flesch Reading Ease : {fre_global:.1f}")
    if   fre_global >= 70: lvl = "Easy - readable by a 6th grade student"
    elif fre_global >= 60: lvl = "Quite easy - middle school level."
    elif fre_global >= 50: lvl = "Quite difficult - high school level."
    elif fre_global >= 30: lvl = "Difficult - university level"
    else:                  lvl = "Very difficult - advanced level."
    print(f" -{lvl}")

    print(f"\nFlesch-Kincaid Grade Level : {fk_global:.1f}")
    if fk_global <= 10:
        print("Computational readability falls at or below approximately Grade 10; this does NOT establish human accessibility or interpretability.")
    elif fk_global <= 13:
        print("Computational readability is approximately at trained/high-school level; human comprehension is not established.")
    else:
        print("Diagnostics are too technical - prompt engineering is recommended.")

    print(f"\nSaved: {output_dir / 'readability_scores_full.csv'}")


if not df_read_cls.empty:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    df_plot = df_read_cls.copy()
    axes[0].bar(df_plot['Class'], df_plot['Flesch RE'], color=['green','blue','orange','red'])
    axes[0].axhline(60, color='gray', linestyle='--', label='Threshold "fairly easy" (60)')
    axes[0].set_title('Flesch Reading Ease by default class')
    axes[0].set_ylabel('Score (higher = more readable)')
    axes[0].legend()
    axes[1].bar(df_plot['Class'], df_plot['FK Grade'], color=['green','blue','orange','red'])
    axes[1].axhline(12, color='gray', linestyle='--', label='High school level (12)')
    axes[1].set_title('Flesch-Kincaid Grade by defect class')
    axes[1].set_ylabel('School grade required.')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(fig_dir / 'readability_by_class.png', dpi=150)
    plt.show()



# # Experiment 2 - Rigorous Cross-Load Generalization 

with open(output_dir/'model_results.json', 'r', encoding='utf-8') as f:
    model_results = json.load(f)

# ## Cross-Load Split: Train on 0-1 HP, Test on 2-3 HP

df['load'] = df['rpm'].apply(cwru_load_from_rpm)
print('Load distribution:')
print(df['load'].value_counts())
print('\nClass distribution by load:')
print(df.groupby(['load', 'label']).size().unstack(fill_value=0))

from sklearn.model_selection import train_test_split

# Strict source/target domains
df_low = df[df['load'].isin(['0 HP', '1 HP'])].copy().reset_index(drop=True)
df_test_exp2 = df[df['load'].isin(['2 HP', '3 HP'])].copy().reset_index(drop=True)

# File-level table and stratify validation files by fault class.
file_meta_low = (
    df_low.groupby('file_num', as_index=False)
          .agg(label=('label','first'), load=('load','first'))
)

low_train_files, low_val_files = train_test_split(
    file_meta_low['file_num'].values,
    test_size=0.20,
    random_state=42,
    stratify=file_meta_low['label'].values,
)

df_tr2 = df_low[df_low['file_num'].isin(low_train_files)].copy().reset_index(drop=True)
df_va2 = df_low[df_low['file_num'].isin(low_val_files)].copy().reset_index(drop=True)

# Leakage checks
assert set(df_tr2['file_num']).isdisjoint(set(df_va2['file_num']))
assert set(df_tr2['file_num']).isdisjoint(set(df_test_exp2['file_num']))
assert set(df_va2['file_num']).isdisjoint(set(df_test_exp2['file_num']))
assert set(df_tr2['load']).issubset({'0 HP','1 HP'})
assert set(df_va2['load']).issubset({'0 HP','1 HP'})
assert set(df_test_exp2['load']).issubset({'2 HP','3 HP'})

print(f'Train low-load: {len(df_tr2)} windows / {df_tr2.file_num.nunique()} files')
print(f'Val   low-load: {len(df_va2)} windows / {df_va2.file_num.nunique()} files')
print(f'Test high-load: {len(df_test_exp2)} windows / {df_test_exp2.file_num.nunique()} files')
print('\nHigh-load test class counts:', df_test_exp2['label'].value_counts().to_dict())

exp2_split_summary = pd.DataFrame([
    {'split':'train_0_1HP','n_files':df_tr2.file_num.nunique(),'n_windows':len(df_tr2)},
    {'split':'val_0_1HP','n_files':df_va2.file_num.nunique(),'n_windows':len(df_va2)},
    {'split':'test_2_3HP','n_files':df_test_exp2.file_num.nunique(),'n_windows':len(df_test_exp2)},
])
exp2_split_summary.to_csv(output_dir / 'exp2_strict_crossload_split.csv', index=False)
display(exp2_split_summary)


# ## RAG and No-RAG Dataset Construction
for d in [df_tr2, df_va2, df_test_exp2]:
    d['text'] = d.apply(encode_to_text, axis=1)
    d['diagnosis'] = d.apply(make_output, axis=1)

rag_corpus_exp2 = df_tr2['text'].tolist()
embedder_exp2 = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0')
emb_exp2 = embedder_exp2.encode(
    rag_corpus_exp2,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True,
).astype('float32')
index_exp2 = faiss.IndexFlatIP(emb_exp2.shape[1])
index_exp2.add(emb_exp2)
print(f'FAISS Exp2 index: {index_exp2.ntotal} LOW-LOAD TRAIN windows only')

def retrieve_exp2(query_text: str, k: int = 3) -> list:
    q = embedder_exp2.encode([query_text], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
    scores, ids = index_exp2.search(q, min(k, index_exp2.ntotal))
    return [{
        'text': df_tr2.iloc[int(i)]['text'],
        'label': df_tr2.iloc[int(i)]['label'],
        'diagnosis': df_tr2.iloc[int(i)]['diagnosis'],
        'file_num': df_tr2.iloc[int(i)]['file_num'],
        'load': df_tr2.iloc[int(i)]['load'],
        'defect_size': df_tr2.iloc[int(i)]['defect_size'],
        'score': float(s),
    } for s, i in zip(scores[0], ids[0])]

def make_dataset_exp2(split_df, use_rag=True, k=3, retrieve_fn=retrieve_exp2):
    samples=[]
    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f'Build Exp2 k={k} RAG={use_rag}'):
        diag=row['diagnosis']; label=row['label']
        sev_m=re.search(r'Severity:\s*(\w+)', diag); severity=sev_m.group(1) if sev_m else 'None'
        exp_m=re.search(r'Explanation:\s*(.+?)(?:\s+Recommendation:)',diag,re.DOTALL)
        explanation=exp_m.group(1).strip() if exp_m else diag
        rec_m=re.search(r'Recommendation:\s*(.+?)(?:\s+Urgency:)',diag,re.DOTALL)
        urg_m=re.search(r'Urgency:\s*(.+)',diag)
        recommendation=(f"{rec_m.group(1).strip()}. Urgency: {urg_m.group(1).strip()}" if rec_m and urg_m
                        else rec_m.group(1).strip() if rec_m else 'Continue routine monitoring.')
        completion=(f'Fault class: {label}\nSeverity: {severity}\nExplanation: {explanation}\nRecommendation: {recommendation}')

        retrieved=[]
        if use_rag and k>0:
            candidates=retrieve_fn(row['text'], k=k+12)
            retrieved=[r for r in candidates if r['file_num'] != row['file_num']][:k]

        samples.append({
            'prompt': build_prompt(row['text'], retrieved if retrieved else None),
            'completion': completion,
            'label': label,
            'severity': severity,
            'file_num': row['file_num'],
            'load': row['load'],
            'text': row['text'],
            'retrieved_labels': [r['label'] for r in retrieved],
            'retrieved_files': [r['file_num'] for r in retrieved],
            'retrieved_loads': [r['load'] for r in retrieved],
        })
    return samples

train_exp2_lm = make_dataset_exp2(df_tr2, use_rag=True, k=3)
val_exp2_lm   = make_dataset_exp2(df_va2, use_rag=True, k=3)
# Complete high-load test set for BOTH conditions.
test_exp2_rag    = make_dataset_exp2(df_test_exp2, use_rag=True,  k=3)
test_exp2_no_rag = make_dataset_exp2(df_test_exp2, use_rag=False, k=0)
assert len(test_exp2_rag) == len(df_test_exp2) == len(test_exp2_no_rag)
print('Complete Exp2 test N =', len(test_exp2_rag))

# ## Dedicated Low-Load LoRA Training

def samples_to_training_dataset(samples):
    ds = Dataset.from_list(samples)
    def _fmt(sample):
        msgs=[
            {'role':'system','content':SYSTEM_PROMPT},
            {'role':'user','content':sample['prompt']},
            {'role':'assistant','content':sample['completion']},
        ]
        return {'text': tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)}
    remove_cols=ds.column_names
    return ds.map(_fmt, remove_columns=remove_cols)

train_exp2_ds = samples_to_training_dataset(train_exp2_lm)
val_exp2_ds   = samples_to_training_dataset(val_exp2_lm)

EXP2_ADAPTER_DIR = output_dir / 'llama32_lora_exp2_lowload_seed42'
# train_lora_adapter(train_exp2_ds, val_exp2_ds, EXP2_ADAPTER_DIR, seed=42, epochs=5)

# ## RAG vs No-RAG Comparison on 2-3 HP

# gc.collect(); torch.cuda.empty_cache()
# exp2_base = AutoModelForCausalLM.from_pretrained(
#     base_model_id,
#     device_map={'': 0},
#     quantization_config=bnb_config,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
#     attn_implementation='eager',
# )
# model_exp2 = PeftModel.from_pretrained(exp2_base, str(EXP2_ADAPTER_DIR))
# model_exp2.eval()
# tok_exp2 = AutoTokenizer.from_pretrained(str(EXP2_ADAPTER_DIR))
# tok_exp2.pad_token = tok_exp2.eos_token

def llm_predict_with(model, tok, prompt_text, max_new_tokens=150):
    msgs=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':prompt_text}]
    enc=tok.apply_chat_template(msgs,add_generation_prompt=True,return_tensors='pt',return_dict=True).to(model.device)
    prompt_len=enc['input_ids'].shape[1]
    with torch.inference_mode():
        out=model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id,
            use_cache=True, repetition_penalty=1.1,
        )
    return tok.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

def run_exp2_inference(samples, condition_name):
    results=[]
    for sample in tqdm(samples, desc=condition_name):
        gen=llm_predict_with(model_exp2,tok_exp2,sample['prompt'])
        results.append({
            'condition':condition_name,
            'true_label':sample['label'],
            'pred_label':parse_class(gen),
            'pred_severity':parse_severity(gen),
            'file_num':int(sample['file_num']),
            'load':sample['load'],
            'retrieved_labels':sample['retrieved_labels'],
            'retrieved_files':sample['retrieved_files'],
            'generated':gen,
            'prompt':sample['prompt'],
        })
    return results

# results_rag = run_exp2_inference(test_exp2_rag, 'RAG_k3')
# results_no_rag = run_exp2_inference(test_exp2_no_rag, 'No_RAG')
# with open(output_dir/'exp2_results_rag_full_seed42.json','w') as f: json.dump(results_rag,f,indent=2)
# with open(output_dir/'exp2_results_no_rag_full_seed42.json','w') as f: json.dump(results_no_rag,f,indent=2)
# print('Saved complete-set paired predictions:', len(results_rag), len(results_no_rag))

# def json_safe(obj):
#     if isinstance(obj, dict):
#         return {k: json_safe(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [json_safe(v) for v in obj]
#     elif isinstance(obj, tuple):
#         return [json_safe(v) for v in obj]
#     elif isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     else:
#         return obj

# results_rag_safe = json_safe(results_rag)
# results_no_rag_safe = json_safe(results_no_rag)

# with open(output_dir/'exp2_results_rag_full_seed42.json','w') as f:
#     json.dump(results_rag_safe, f, indent=2)

# with open(output_dir/'exp2_results_no_rag_full_seed42.json','w') as f:
#     json.dump(results_no_rag_safe, f, indent=2)

# print('Saved complete-set paired predictions:',
#       len(results_rag_safe),
#       len(results_no_rag_safe))

with open(output_dir / 'exp2_results_rag_full_seed42.json', 'r', encoding='utf-8') as f:
    results_rag = json.load(f)

with open(output_dir / 'exp2_results_no_rag_full_seed42.json', 'r', encoding='utf-8') as f:
    results_no_rag = json.load(f)

def eval_results(results: list, name: str) -> dict:
    # Unknown outputs count as errors rather than silently disappearing.
    y_true=[r['true_label'] for r in results]
    y_pred=[r['pred_label'] for r in results]
    labels=['Normal','Ball','Inner Race','Outer Race']
    parse_rate=np.mean([p!='Unknown' for p in y_pred])
    # sklearn can include Unknown as an extra predicted label; fixed labels keep the target-class macro metrics.
    acc=accuracy_score(y_true,y_pred)
    prec=precision_score(y_true,y_pred,labels=labels,average='macro',zero_division=0)
    rec=recall_score(y_true,y_pred,labels=labels,average='macro',zero_division=0)
    f1=f1_score(y_true,y_pred,labels=labels,average='macro',zero_division=0)
    print(f'\n{name} N={len(results)} parse={parse_rate*100:.2f}%')
    print(f'Accuracy={acc*100:.2f}% Precision={prec*100:.2f}% Recall={rec*100:.2f}% Macro-F1={f1*100:.2f}%')
    print(classification_report(y_true,y_pred,labels=labels,zero_division=0))
    return {'model':name,'accuracy':acc*100,'precision':prec*100,'recall':rec*100,'f1_score':f1*100,
            'parse_rate':parse_rate*100,'y_true':y_true,'y_pred':y_pred}

res_exp2_rag=eval_results(results_rag,'LLaMA 3.2 + low-load LoRA + RAG k=3')
res_exp2_no_rag=eval_results(results_no_rag,'LLaMA 3.2 + low-load LoRA, no RAG')

df_low_dev=pd.concat([df_tr2,df_va2],ignore_index=True)
X_low=df_low_dev[feature_names].values; y_low=df_low_dev['y'].values; g_low=df_low_dev['file_num'].values

# SVM GroupKFold source-domain tuning
svm_grid={'C':[0.1,1,10,100,1000],'gamma':['scale',0.01,0.001,0.0001]}
best_svm=(-np.inf,None)
for params in ParameterGrid(svm_grid):
    scores=[]
    for tr,va in GroupKFold(n_splits=4).split(X_low,y_low,groups=g_low):
        sc=StandardScaler(); Xtr=sc.fit_transform(X_low[tr]); Xva=sc.transform(X_low[va])
        m=SVC(kernel='rbf',**params); m.fit(Xtr,y_low[tr]); scores.append(f1_score(y_low[va],m.predict(Xva),average='macro'))
    if np.mean(scores)>best_svm[0]: best_svm=(np.mean(scores),params)
print('Exp2 best SVM:',best_svm)

scaler_exp2=StandardScaler(); X_low_sc=scaler_exp2.fit_transform(X_low); X_high_sc=scaler_exp2.transform(df_test_exp2[feature_names].values)
svm_exp2=SVC(kernel='rbf',**best_svm[1]); svm_exp2.fit(X_low_sc,y_low)
y_svm=svm_exp2.predict(X_high_sc)
res_svm_exp2={'model':'SVM source-domain tuned','accuracy':accuracy_score(df_test_exp2.y,y_svm)*100,
              'precision':precision_score(df_test_exp2.y,y_svm,average='macro',zero_division=0)*100,
              'recall':recall_score(df_test_exp2.y,y_svm,average='macro',zero_division=0)*100,
              'f1_score':f1_score(df_test_exp2.y,y_svm,average='macro',zero_division=0)*100,
              'y_true':df_test_exp2.y.tolist(),'y_pred':y_svm.tolist()}

# RF GroupKFold source-domain tuning
rf_grid={'n_estimators':[200,300,500],'max_depth':[None,10,20,30]}
best_rf=(-np.inf,None)
for params in ParameterGrid(rf_grid):
    scores=[]
    for tr,va in GroupKFold(n_splits=4).split(X_low,y_low,groups=g_low):
        m=RandomForestClassifier(**params,max_features='sqrt',random_state=42,n_jobs=-1)
        m.fit(X_low[tr],y_low[tr]); scores.append(f1_score(y_low[va],m.predict(X_low[va]),average='macro'))
    if np.mean(scores)>best_rf[0]: best_rf=(np.mean(scores),params)
print('Exp2 best RF:',best_rf)
rf_exp2=RandomForestClassifier(**best_rf[1],max_features='sqrt',random_state=42,n_jobs=-1)
rf_exp2.fit(X_low,y_low); y_rf=rf_exp2.predict(df_test_exp2[feature_names].values)
res_rf_exp2={'model':'RF source-domain tuned','accuracy':accuracy_score(df_test_exp2.y,y_rf)*100,
             'precision':precision_score(df_test_exp2.y,y_rf,average='macro',zero_division=0)*100,
             'recall':recall_score(df_test_exp2.y,y_rf,average='macro',zero_division=0)*100,
             'f1_score':f1_score(df_test_exp2.y,y_rf,average='macro',zero_division=0)*100,
             'y_true':df_test_exp2.y.tolist(),'y_pred':y_rf.tolist()}

print('\nSVM', {k:v for k,v in res_svm_exp2.items() if k not in ['y_true','y_pred']})
print('RF ', {k:v for k,v in res_rf_exp2.items() if k not in ['y_true','y_pred']})

df_exp2=pd.DataFrame([{
    'Model':r['model'],'Accuracy':r['accuracy'],'Precision':r['precision'],'Recall':r['recall'],'F1 Score':r['f1_score']
} for r in [res_svm_exp2,res_rf_exp2,res_exp2_no_rag,res_exp2_rag]])
print(df_exp2.to_string(index=False))
df_exp2.to_csv(output_dir/'exp2_results_full_strict_seed42.csv',index=False)

label_names_order=['Normal','Ball','Inner Race','Outer Race']
class_reports={}
for name,yt,yp in [
    ('LLM_RAG',res_exp2_rag['y_true'],res_exp2_rag['y_pred']),
    ('LLM_NoRAG',res_exp2_no_rag['y_true'],res_exp2_no_rag['y_pred']),
    ('SVM',[label_names[i] for i in res_svm_exp2['y_true']],[label_names[i] for i in res_svm_exp2['y_pred']]),
    ('RF',[label_names[i] for i in res_rf_exp2['y_true']],[label_names[i] for i in res_rf_exp2['y_pred']]),
]:
    rep=pd.DataFrame(classification_report(yt,yp,labels=label_names_order,output_dict=True,zero_division=0)).T
    rep.to_csv(output_dir/f'exp2_class_report_{name}.csv')
    class_reports[name]=rep
    print('\n',name); display(rep.loc[label_names_order,['precision','recall','f1-score','support']])

# # Confusion matrices + seed-42 paired statistical comparison.
# plot_cm([label_map[x] for x in res_exp2_rag['y_true']],
#         [label_map.get(x,-1) for x in res_exp2_rag['y_pred']],
#         'Strict cross-load LLM + RAG',fig_dir/'exp2_revision_llm_rag_cm.png')
# plot_cm([label_map[x] for x in res_exp2_no_rag['y_true']],
#         [label_map.get(x,-1) for x in res_exp2_no_rag['y_pred']],
#         'Strict cross-load LLM no RAG',fig_dir/'exp2_revision_llm_no_rag_cm.png')
# plot_cm(res_svm_exp2['y_true'],res_svm_exp2['y_pred'],'Strict cross-load SVM',fig_dir/'exp2_revision_svm_cm.png')
# plot_cm(res_rf_exp2['y_true'],res_rf_exp2['y_pred'],'Strict cross-load RF',fig_dir/'exp2_revision_rf_cm.png')

# # McNemar exact test on paired correctness (same 1,961 windows).
# from scipy.stats import binomtest
# paired=pd.DataFrame({
    
#     'true':[r['true_label'] for r in results_rag],
#     'rag':[r['pred_label'] for r in results_rag],
#     'norag':[r['pred_label'] for r in results_no_rag],
#     'file_num':[r['file_num'] for r in results_rag],
# })
# paired['rag_correct']=paired['true']==paired['rag']
# paired['norag_correct']=paired['true']==paired['norag']
# b=int(((paired.rag_correct==1)&(paired.norag_correct==0)).sum())
# c=int(((paired.rag_correct==0)&(paired.norag_correct==1)).sum())
# p_mcnemar=binomtest(min(b,c),n=b+c,p=0.5,alternative='two-sided').pvalue if (b+c)>0 else 1.0
# print(f'McNemar exact discordant pairs: RAG-only correct={b}, NoRAG-only correct={c}, p={p_mcnemar:.6g}')

# # Cluster bootstrap by recording file to respect dependence among overlapping windows.
# def cluster_bootstrap_f1_delta(df_pairs,n_boot=5000,seed=2026):
#     rng=np.random.default_rng(seed); files=df_pairs.file_num.unique(); vals=[]
#     for _ in range(n_boot):
#         sampled=rng.choice(files,size=len(files),replace=True)
#         parts=[]
#         for j,f in enumerate(sampled):
#             z=df_pairs[df_pairs.file_num==f].copy(); z['_boot_cluster']=j; parts.append(z)
#         boot=pd.concat(parts,ignore_index=True)
#         f_r=f1_score(boot.true,boot.rag,labels=label_names_order,average='macro',zero_division=0)
#         f_n=f1_score(boot.true,boot.norag,labels=label_names_order,average='macro',zero_division=0)
#         vals.append(f_r-f_n)
#     vals=np.asarray(vals)
#     return {'mean_delta':vals.mean(),'ci_low':np.quantile(vals,.025),'ci_high':np.quantile(vals,.975),'p_delta_le_0':np.mean(vals<=0)}

# boot_stats=cluster_bootstrap_f1_delta(paired)
# print('Cluster-bootstrap Δmacro-F1 (RAG - noRAG):',boot_stats)
# with open(output_dir/'exp2_seed42_statistical_tests.json','w') as f:
#     json.dump({'mcnemar_b':b,'mcnemar_c':c,'mcnemar_p':p_mcnemar,'cluster_bootstrap':boot_stats},f,indent=2)


# # Experiment 3 : Paderborn KAt Dataset
# 
# 
# 
# **URL:** https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter/data-sets-and-download
# 

# ## Paderborn Paths and Configuration

# In[62]:


FS_PADERBORN = 64_000

PAD_WINDOW_SIZE = window_size
PAD_WINDOW_STEP = window_step

MAX_FILES_PER_CLASS = 60
MAX_WINDOWS_PER_FILE = 25

PADERBORN_CHANNEL_INDEX = 6  

general_feature_names = [
    "rms",
    "kurt",
    "skewness",
    "peak",
    "crest_factor",
    "std",
    "mav",
    "dominant_freq",
    "spec_centroid",
    "spec_std"
]

label_map = {
    "Normal": 0,
    "Inner Race": 1,
    "Ball": 2,
    "Outer Race": 3
}

label_names = ["Normal", "Inner Race", "Ball", "Outer Race"]

pad_data_dir = base_dir / "paderborn_data"
pad_output_dir = output_dir / "paderborn_exp4"
pad_fig_dir = pad_output_dir / "figures"
pad_llm_dir = pad_output_dir / "llm_paderborn_rag"

#pad_output_dir.mkdir(parents=True, exist_ok=True)
#pad_fig_dir.mkdir(parents=True, exist_ok=True)


for d in [pad_output_dir, pad_fig_dir, pad_llm_dir]:
    d.mkdir(parents=True, exist_ok=True)


# In[63]:


PAD_CONDITION_MAP = {
    "N15_M07_F10": {"rpm": 1500.0, "torque_nm": 0.7, "force_n": 1000},
    "N09_M07_F10": {"rpm":  900.0, "torque_nm": 0.7, "force_n": 1000},
    "N15_M01_F10": {"rpm": 1500.0, "torque_nm": 0.1, "force_n": 1000},
    "N15_M07_F04": {"rpm": 1500.0, "torque_nm": 0.7, "force_n": 400},
}


# ##  Paderborn MATLAB File Loader

# In[64]:


def get_pad_label_from_folder(folder_name: str) -> str | None:
    """
    Convert Paderborn folder name to class label.
    Examples:
      K001 -> Normal
      KA04 -> Outer Race
      KI14 -> Inner Race
      KB23 -> Ball
    """
    if folder_name.startswith("K0"):
        return "Normal"
    if folder_name.startswith("KA"):
        return "Outer Race"
    if folder_name.startswith("KI"):
        return "Inner Race"
    if folder_name.startswith("KB"):
        return "Ball"
    return None


def parse_pad_condition(filename: str) -> dict:
    """
    Parse filename such as:
      N15_M07_F10_KA08_1.mat
    """
    stem = Path(filename).stem

    m = re.search(r"(N\d+_M\d+_F\d+)", stem)
    condition_key = m.group(1) if m else None

    condition = PAD_CONDITION_MAP.get(
        condition_key,
        {"rpm": np.nan, "torque_nm": np.nan, "force_n": np.nan}
    )

    rpm = condition["rpm"]
    fr = rpm / 60.0 if not np.isnan(rpm) else np.nan

    return {
        "condition": condition_key,
        "rpm": rpm,
        "fr": fr,
        "torque_nm": condition["torque_nm"],
        "force_n": condition["force_n"],
    }

def extract_vibration_from_paderborn_mat(fp: Path, channel_index=6, verbose=False):
    mat = loadmat(fp, squeeze_me=True, struct_as_record=False)
    stem = fp.stem

    root = mat[stem]
    Y = root.Y
    sig = np.asarray(Y[channel_index].Data).squeeze().astype(np.float32)

    if sig.ndim != 1 or len(sig) < 1000:
        raise ValueError(f"Invalid signal in {fp.name}: shape={sig.shape}")

    return sig

sample_file = sorted(pad_data_dir.rglob("*.mat"))[0]

sig = extract_vibration_from_paderborn_mat(
    sample_file,
    channel_index=6,
    verbose=True
)

print("Shape:", sig.shape)
print("Mean:", sig.mean())
print("Std:", sig.std())

from collections import defaultdict
def collect_paderborn_files_by_class(pad_data_dir, max_files_per_class=MAX_FILES_PER_CLASS, random_state=42):
    random.seed(random_state)

    files_by_class = defaultdict(list)

    for fp in sorted(pad_data_dir.rglob("*.mat")):
        folder = fp.parent.name
        label = get_pad_label_from_folder(folder)

        if label is not None:
            files_by_class[label].append(fp)

    selected_files = []

    for label, files in files_by_class.items():
        random.shuffle(files)
        selected = files[:max_files_per_class]
        selected_files.extend(selected)
        print(f"{label}: selected {len(selected)} / {len(files)} files")

    return sorted(selected_files)


selected_pad_files = collect_paderborn_files_by_class(
    pad_data_dir,
    max_files_per_class=MAX_FILES_PER_CLASS,
    random_state=42
)

print("Total selected files:", len(selected_pad_files))


def load_paderborn_recording(fp: Path):
    folder_name = fp.parent.name
    label = get_pad_label_from_folder(folder_name)

    if label is None:
        return None

    try:
        sig = extract_vibration_from_paderborn_mat(fp, channel_index=6)
    except Exception as e:
        print(f"Error loading {fp.name}: {e}")
        return None

    cond = parse_pad_condition(fp.name)

    if np.isnan(cond["rpm"]):
        print(f"Warning: RPM not found for {fp.name}. Skipping.")
        return None

    return {
        "file_id": f"{folder_name}_{fp.stem}",
        "folder": folder_name,
        "filename": fp.name,
        "label": label,
        "defect_size": np.nan,
        "signal": sig,
        "rpm": cond["rpm"],
        "fr": cond["fr"],
        "condition": cond["condition"],
        "torque_nm": cond["torque_nm"],
        "force_n": cond["force_n"],
        "n_samples": len(sig),
        "dataset": "Paderborn",
    }

paderborn_records = []

for fp in tqdm(selected_pad_files, desc="Loading selected Paderborn files"):
    rec = load_paderborn_recording(fp)

    if rec is not None:
        paderborn_records.append(rec)

print(f"{len(paderborn_records)} Paderborn files loaded")

df_pad_meta = pd.DataFrame([
    {k: v for k, v in r.items() if k != "signal"}
    for r in paderborn_records
])

print(df_pad_meta["label"].value_counts())

from scipy.signal import butter, sosfilt

features_cache = pad_output_dir / "df_pad_features.parquet"

if features_cache.exists():
    df_pad = pd.read_parquet(features_cache)
    print(f"✓ Loaded df_pad from cache ({features_cache.stat().st_size / 1e6:.1f} MB)")
    print(f"  Shape: {df_pad.shape}")
    print(df_pad["label"].value_counts())
else:
    print("No cache — extracting features from raw signals...")
    rows_pad = []

    for rec in tqdm(paderborn_records, desc="Paderborn feature extraction"):
        windows = segment(
            rec["signal"],
            L=PAD_WINDOW_SIZE,
            S=PAD_WINDOW_STEP
        )

        if len(windows) > MAX_WINDOWS_PER_FILE:
            idx = np.linspace(0, len(windows) - 1, MAX_WINDOWS_PER_FILE).astype(int)
            windows = windows[idx]

        for w_id, window in enumerate(windows):
            features = extract_features(
                window,
                fr=rec["fr"],
                fs=64000,
                dataset="Paderborn"
            )

            row = {name: val for name, val in zip(feature_names, features)}

            row.update({
                "label": rec["label"],
                "defect_size": rec["defect_size"],
                "rpm": rec["rpm"],
                "fr": rec["fr"],
                "file_num": rec["file_id"],
                "folder": rec["folder"],
                "filename": rec["filename"],
                "condition": rec["condition"],
                "torque_nm": rec["torque_nm"],
                "force_n": rec["force_n"],
                "window_id": w_id,
                "dataset": "Paderborn",
            })

            rows_pad.append(row)

    df_pad = pd.DataFrame(rows_pad)
    df_pad["y"] = df_pad["label"].map(label_map)

    # Sauvegarde
    df_pad.to_parquet(features_cache, index=False)
    print(f" Saved {len(df_pad)} feature rows to {features_cache}")
    print(f"  Size: {features_cache.stat().st_size / 1e6:.1f} MB")
    print(f"  Shape: {df_pad.shape}")
    print(df_pad["label"].value_counts())
    print("Missing labels:", df_pad["y"].isna().sum())


# ## Paderborn File-Level Train / Validation / Test Split

from sklearn.model_selection import GroupShuffleSplit

groups = df_pad["file_num"].values


gss1 = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
trainval_idx, test_idx = next(gss1.split(df_pad, df_pad["y"], groups=groups))

df_pad_trainval = df_pad.iloc[trainval_idx].reset_index(drop=True)
df_pad_test = df_pad.iloc[test_idx].reset_index(drop=True)


groups_trainval = df_pad_trainval["file_num"].values

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
train_idx, val_idx = next(
    gss2.split(df_pad_trainval, df_pad_trainval["y"], groups=groups_trainval)
)

df_pad_train = df_pad_trainval.iloc[train_idx].reset_index(drop=True)
df_pad_val = df_pad_trainval.iloc[val_idx].reset_index(drop=True)

print("Train:", df_pad_train.shape)
print("Val  :", df_pad_val.shape)
print("Test :", df_pad_test.shape)

print("\nTrain distribution:")
print(df_pad_train["label"].value_counts())

print("\nVal distribution:")
print(df_pad_val["label"].value_counts())

print("\nTest distribution:")
print(df_pad_test["label"].value_counts())

train_files = set(df_pad_train["file_num"].unique())
val_files = set(df_pad_val["file_num"].unique())
test_files = set(df_pad_test["file_num"].unique())

print("\nLeakage checks:")
print("Train ∩ Val :", len(train_files & val_files))
print("Train ∩ Test:", len(train_files & test_files))
print("Val ∩ Test  :", len(val_files & test_files))

X_pad_train = df_pad_train[feature_names].values
y_pad_train = df_pad_train["y"].values

X_pad_val = df_pad_val[feature_names].values
y_pad_val = df_pad_val["y"].values

X_pad_test = df_pad_test[feature_names].values
y_pad_test = df_pad_test["y"].values


pad_scaler = StandardScaler()

X_pad_train_sc = pad_scaler.fit_transform(X_pad_train)
X_pad_val_sc = pad_scaler.transform(X_pad_val)
X_pad_test_sc = pad_scaler.transform(X_pad_test)


print(f'X_train_sc : {X_pad_train_sc.shape}')
print(f'X_val_sc   : {X_pad_val_sc.shape}')
print(f'X_test_sc  : {X_pad_test_sc.shape}')

svm_pad = SVC(
    kernel="rbf",
    C=10,
    gamma=0.01,
    random_state=42
)

svm_pad.fit(X_pad_train_sc, y_pad_train)

svm_pad_val_results = evaluate(
    svm_pad,
    X_pad_val_sc,
    y_pad_val,
    "SVM - Paderborn Val"
)

svm_pad_test_results = evaluate(
    svm_pad,
    X_pad_test_sc,
    y_pad_test,
    "SVM - Paderborn Test"
)

plot_cm(
    svm_pad_test_results["y_true"],
    svm_pad_test_results["y_pred"],
    "SVM - Paderborn Test",
    pad_fig_dir / "svm_paderborn_test_cm.png"
)

rf_pad = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf_pad.fit(X_pad_train, y_pad_train)

rf_pad_val_results = evaluate(
    rf_pad,
    X_pad_val,
    y_pad_val,
    "Random Forest - Paderborn Val"
)

rf_pad_test_results = evaluate(
    rf_pad,
    X_pad_test,
    y_pad_test,
    "Random Forest - Paderborn Test"
)

plot_cm(
    rf_pad_test_results["y_true"],
    rf_pad_test_results["y_pred"],
    "Random Forest - Paderborn Test",
    pad_fig_dir / "rf_paderborn_test_cm.png"
)
pad_baseline_summary = pd.DataFrame([
    {
        "Experiment": " Paderborn",
        "Model": "SVM",
        "Split": "Validation",
        "Accuracy": svm_pad_val_results["accuracy"],
        "Precision": svm_pad_val_results["precision"],
        "Recall": svm_pad_val_results["recall"],
        "F1-score": svm_pad_val_results["f1_score"],
    },
    {
        "Experiment": " Paderborn",
        "Model": "SVM",
        "Split": "Test",
        "Accuracy": svm_pad_test_results["accuracy"],
        "Precision": svm_pad_test_results["precision"],
        "Recall": svm_pad_test_results["recall"],
        "F1-score": svm_pad_test_results["f1_score"],
    },
    {
        "Experiment": " Paderborn",
        "Model": "Random Forest",
        "Split": "Validation",
        "Accuracy": rf_pad_val_results["accuracy"],
        "Precision": rf_pad_val_results["precision"],
        "Recall": rf_pad_val_results["recall"],
        "F1-score": rf_pad_val_results["f1_score"],
    },
    {
        "Experiment": " Paderborn",
        "Model": "Random Forest",
        "Split": "Test",
        "Accuracy": rf_pad_test_results["accuracy"],
        "Precision": rf_pad_test_results["precision"],
        "Recall": rf_pad_test_results["recall"],
        "F1-score": rf_pad_test_results["f1_score"],
    },
])

pad_baseline_summary.to_csv(
    pad_output_dir / "paderborn_baseline_summary.csv",
    index=False
)
pad_baseline_summary

correct_dominance = {'Normal': 0, 'Inner Race': 0, 'Outer Race': 0, 'Ball': 0}
total = {'Normal': 0, 'Inner Race': 0, 'Outer Race': 0, 'Ball': 0}

expected_dominant = {
    'Inner Race': 'bpfi_ratio',
    'Outer Race': 'bpfo_ratio',
    'Ball': 'bsf_ratio',
}

for _, row in df_pad_train.iterrows():
    cls = row['label']
    total[cls] += 1
    ratios = {
        'bpfi_ratio': row['bpfi_ratio'],
        'bpfo_ratio': row['bpfo_ratio'],
        'bsf_ratio':  row['bsf_ratio'],
    }
    dominant_band = max(ratios, key=ratios.get)
    if cls == 'Normal':
        if max(ratios.values()) < 0.005:
            correct_dominance[cls] += 1
    else:
        if dominant_band == expected_dominant[cls]:
            correct_dominance[cls] += 1

print(f"{'Class':12s} {'Correct':>8s} / {'Total':>6s}  {'%':>6s}")
print("-" * 45)
for cls in ['Normal', 'Inner Race', 'Outer Race', 'Ball']:
    pct = 100 * correct_dominance[cls] / total[cls] if total[cls] > 0 else 0
    print(f"{cls:12s} {correct_dominance[cls]:>8d} / {total[cls]:>6d}  {pct:>5.1f}%")

importances = pd.DataFrame({
    'feature':    feature_names,
    'importance': rf_pad.feature_importances_,
}).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(importances['feature'], importances['importance'], color='steelblue')
ax.set_title('Random Forest - Feature Importance')
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(fig_dir / 'rf_feature_importance.png', dpi=150)
plt.show()

# ## Paderborn Sensor-to-Text Encoding

def encode_to_text_paderborn(row):

    ff = fault_frequencies(row['fr'], dataset='Paderborn')

    return (
        f"Dataset: Paderborn (FAG 6203 bearing).\n"
        f"Operating condition: {row['condition']} "
        f"(speed={row['rpm']:.0f} RPM, fr={row['fr']:.2f} Hz, "
        f"torque={row['torque_nm']} Nm, force={row['force_n']} N).\n"
        f"Expected fault frequencies: "
        f"BPFI={ff['f_BPFI']:.1f} Hz, BPFO={ff['f_BPFO']:.1f} Hz, BSF={ff['f_BSF']:.1f} Hz.\n"
        f"Vibration features:\n"
        f"- RMS: {row['rms']:.4f}\n"
        f"- Kurtosis: {row['kurt']:.4f}\n"
        f"- Skewness: {row['skewness']:.4f}\n"
        f"- Peak: {row['peak']:.4f}\n"
        f"- Crest factor: {row['crest_factor']:.4f}\n"
        f"- Std: {row['std']:.4f}\n"
        f"- MAV: {row['mav']:.4f}\n"
        f"- Dominant frequency: {row['dominant_freq']:.2f} Hz\n"
        f"- Spectral centroid: {row['spec_centroid']:.2f} Hz\n"
        f"- BPFO band energy: {row['bpfo_energy']:.3e}\n"
        f"- BPFI band energy: {row['bpfi_energy']:.3e}\n"
        f"- BSF band energy: {row['bsf_energy']:.3e}\n"
        f"- BPFO ratio: {row['bpfo_ratio']:.6f}\n"
        f"- BPFI ratio: {row['bpfi_ratio']:.6f}\n"
        f"- BSF ratio: {row['bsf_ratio']:.6f}\n"
        f"- Residual ratio: {row['residual_ratio']:.6f}"
    )


def make_paderborn_diagnosis(row):
    label = row["label"]
    ff = fault_frequencies(row['fr'], dataset='Paderborn')
    bpfo_r = row['bpfo_ratio']
    bpfi_r = row['bpfi_ratio']
    bsf_r  = row['bsf_ratio']
    res_r  = row['residual_ratio']

    ratios_str = (
        f"BPFI ratio={bpfi_r:.5f}, BPFO ratio={bpfo_r:.5f}, "
        f"BSF ratio={bsf_r:.5f}, residual ratio={res_r:.5f}"
    )

    if label == "Normal":
        return (
            f"Fault class: Normal\n"
            f"Severity: None\n"
            f"Explanation: All three fault-band ratios are comparable in magnitude "
            f"({ratios_str}), with no single band dominating. Kurtosis={row['kurt']:.2f} "
            f"reflects baseline mechanical noise rather than a defect signature. "
            f"Residual ratio {res_r:.4f} indicates that most spectral energy lies "
            f"outside fault bands, consistent with healthy operation.\n"
            f"Recommendation: Continue routine monitoring."
        )

    fault_data = {
        'Inner Race': ('BPFI', bpfi_r, ff['f_BPFI'], 'inner raceway',
                       f"BPFI ratio ({bpfi_r:.5f}) is the dominant fault band, "
                       f"exceeding BPFO ({bpfo_r:.5f}) and BSF ({bsf_r:.5f})"),
        'Outer Race': ('BPFO', bpfo_r, ff['f_BPFO'], 'outer raceway',
                       f"BPFO ratio ({bpfo_r:.5f}) is the dominant fault band, "
                       f"exceeding BPFI ({bpfi_r:.5f}) and BSF ({bsf_r:.5f})"),
        'Ball':       ('BSF',  bsf_r, ff['f_BSF'],  'rolling elements',
                       f"BSF ratio ({bsf_r:.5f}) is elevated relative to "
                       f"BPFO ({bpfo_r:.5f}) and BPFI ({bpfi_r:.5f})"),
    }
    band_name, _, band_freq, surface, comparison = fault_data[label]
    return (
        f"Fault class: {label}\n"
        f"Severity: Medium\n"
        f"Explanation: {comparison} at expected frequency {band_freq:.1f} Hz. "
        f"Kurtosis={row['kurt']:.2f}. This pattern indicates impulsive defect "
        f"on the {surface}.\n"
        f"Recommendation: Inspect the {surface} and plan bearing replacement."
    )

for d in [df_pad_train, df_pad_val, df_pad_test]:
    d["text"] = d.apply(encode_to_text_paderborn, axis=1)
    d["diagnosis"] = d.apply(make_paderborn_diagnosis, axis=1)

print(df_pad_train[["label", "text", "diagnosis"]].head(2).to_string())

print("Ratios moyens par classe (Paderborn train, après filtrage)\n")
print(f"{'Classe':12s} {'BPFO':>10s} {'BPFI':>10s} {'BSF':>10s} {'Kurt':>8s}")
print("-" * 52)
for cls in ['Normal', 'Inner Race', 'Outer Race', 'Ball']:
    sub = df_pad_train[df_pad_train['label'] == cls]
    if len(sub) == 0: continue
    bpfo = sub['bpfo_ratio'].mean()
    bpfi = sub['bpfi_ratio'].mean()
    bsf  = sub['bsf_ratio'].mean()
    kurt = sub['kurt'].mean()
    ratios = {'BPFO': bpfo, 'BPFI': bpfi, 'BSF': bsf}
    dominant = max(ratios, key=ratios.get)
    print(f"{cls:12s} {bpfo:>10.6f} {bpfi:>10.6f} {bsf:>10.6f} {kurt:>8.2f}  -> {dominant}")

embedder_pad = SentenceTransformer("all-MiniLM-L6-v2",device=("cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"))

pad_train_texts = df_pad_train["text"].tolist()

pad_train_embeddings = embedder_pad.encode(
    pad_train_texts,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
).astype("float32")

dim = pad_train_embeddings.shape[1]

pad_index = faiss.IndexFlatIP(dim)
pad_index.add(pad_train_embeddings)

print("Embedding shape:", pad_train_embeddings.shape)
print("FAISS index size:", pad_index.ntotal)

def retrieve_pad(query_text, k=3):
    q_emb = embedder_pad.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = pad_index.search(q_emb, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        row = df_pad_train.iloc[int(idx)]

        results.append({
            "score": float(score),
            "label": row["label"],
            "file_num": row["file_num"],
            "text": row["text"],
            "diagnosis": row["diagnosis"]
        })

    return results

def retrieval_hit_at_k_pad(df_subset, k=3, n_samples=200):
    eval_df = df_subset.sample(min(n_samples, len(df_subset)), random_state=42)
    hits = 0
    hits_by_class = {'Normal': [0,0], 'Inner Race': [0,0], 'Outer Race': [0,0], 'Ball': [0,0]}

    for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Hit@k filtered"):
        retrieved_all = retrieve_pad(row["text"], k=k + 5)
        retrieved_filtered = [
            r for r in retrieved_all if r['file_num'] != row['file_num']
        ][:k]
        retrieved_labels = [r["label"] for r in retrieved_filtered]
        hits_by_class[row['label']][1] += 1
        if row["label"] in retrieved_labels:
            hits += 1
            hits_by_class[row['label']][0] += 1

    print(f"\nHit@3 global: {hits/len(eval_df)*100:.1f}%")
    print(f"\nHit@3 par classe:")
    for cls, (h, t) in hits_by_class.items():
        if t > 0:
            print(f"  {cls:12s}: {h/t*100:5.1f}%  ({h}/{t})")
    return hits / len(eval_df)


hit_train = retrieval_hit_at_k_pad(df_pad_train, k=3, n_samples=200)
print(f"\n TEST ")
hit_test  = retrieval_hit_at_k_pad(df_pad_test,  k=3, n_samples=200)


# ## Paderborn RAG Prompt Construction

RAG_K = 3
MAX_LENGTH = 1536

SYSTEM_PROMPT_PAD = (
    "You are an expert industrial maintenance assistant specialized in rolling-element "
    "bearing fault diagnosis using vibration signals from the Paderborn bearing dataset. "
    "Given vibration-derived features and similar retrieved historical cases, produce a "
    "structured diagnostic report with fault class, severity, explanation, and maintenance recommendation."
)


def compact_case_text(row):
    """Compact representation INCLUDING fault-band energies, ratios, and expected frequencies.
    Without these, the LLM cannot compare measured vs expected fault signatures.
    """
    ff = fault_frequencies(row['fr'], dataset='Paderborn')
    return (
        f"condition={row['condition']}, "
        f"rpm={row['rpm']:.0f}, fr={row['fr']:.2f} Hz, "
        f"expected: BPFI={ff['f_BPFI']:.1f} Hz, BPFO={ff['f_BPFO']:.1f} Hz, BSF={ff['f_BSF']:.1f} Hz, "
        f"rms={row['rms']:.4f}, "
        f"kurt={row['kurt']:.4f}, "
        f"skewness={row['skewness']:.4f}, "
        f"peak={row['peak']:.4f}, "
        f"crest_factor={row['crest_factor']:.4f}, "
        f"std={row['std']:.4f}, "
        f"mav={row['mav']:.4f}, "
        f"dominant_freq={row['dominant_freq']:.2f} Hz, "
        f"spec_centroid={row['spec_centroid']:.2f} Hz, "
        f"BPFO_energy={row['bpfo_energy']:.3e}, "
        f"BPFI_energy={row['bpfi_energy']:.3e}, "
        f"BSF_energy={row['bsf_energy']:.3e}, "
        f"BPFO_ratio={row['bpfo_ratio']:.5f}, "
        f"BPFI_ratio={row['bpfi_ratio']:.5f}, "
        f"BSF_ratio={row['bsf_ratio']:.5f}, "
        f"residual_ratio={row['residual_ratio']:.5f}"
    )


def retrieve_pad_compact(query_text, k=3):
    q_emb = embedder_pad.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = pad_index.search(q_emb, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        row = df_pad_train.iloc[int(idx)]
        results.append({
            "score": float(score),
            "label": row["label"],
            "file_num": row["file_num"],
            "features": compact_case_text(row),   # ← maintenant inclut tout
            "diagnosis": row["diagnosis"]
        })
    return results


def build_prompt_pad_compact(row, retrieved=None):
    context = ""
    if retrieved:
        context_lines = []
        for i, r in enumerate(retrieved, 1):
            context_lines.append(
                f"[Retrieved case {i}] "
                f"score={r['score']:.4f}, "
                f"label={r['label']}, "
                f"features: {r['features']}\n"
                f"Diagnosis: {r['diagnosis']}"
            )
        context = (
            "Similar historical cases retrieved from the training set:\n"
            + "\n".join(context_lines)
            + "\n\n"
        )

    current_case = compact_case_text(row)

    prompt = (
        f"{context}"
        f"Current case:\n"
        f"Dataset=Paderborn, channel=vibration_1, features: {current_case}\n\n"
        f"Task:\n"
        f"Diagnose the current bearing condition. Use the retrieved cases as supporting evidence, "
        f"but compare them with the current vibration features. "
        f"Pay particular attention to which fault band (BPFI, BPFO, or BSF) has the highest energy/ratio.\n\n"
        f"Return exactly this format:\n"
        f"Fault class: [Normal | Inner Race | Ball | Outer Race]\n"
        f"Severity: [None | Low | Medium | High]\n"
        f"Explanation: [short signal-based explanation]\n"
        f"Recommendation: [maintenance action and urgency]"
    )

    return prompt


# ## Paderborn LLM Dataset Construction

def make_dataset_pad_compact(split_df, use_rag=True, n_max=None, k=3):
    if n_max is not None:
        split_df = split_df.sample(
            min(n_max, len(split_df)),
            random_state=42
        )

    samples = []
    for _, row in tqdm(split_df.iterrows(), total=len(split_df),
                       desc="Building compact Paderborn LLM dataset"):
        retrieved = None
        retrieved_labels = []
        retrieved_scores = []

        if use_rag:
            # Récupérer k+5 pour avoir de la marge après filtrage
            retrieved_all = retrieve_pad_compact(row["text"], k=k + 5)
            # Filtrer : exclure les cas du même fichier
            retrieved_filtered = [
                r for r in retrieved_all if r['file_num'] != row['file_num']
            ][:k]
            retrieved = retrieved_filtered
            retrieved_labels = [r["label"] for r in retrieved]
            retrieved_scores = [r["score"] for r in retrieved]

        prompt = build_prompt_pad_compact(row, retrieved)

        samples.append({
            "prompt": prompt,
            "completion": row["diagnosis"],
            "label": row["label"],
            "file_num": row["file_num"],
            "retrieved_labels": retrieved_labels,
            "retrieved_scores": retrieved_scores,
        })

    return samples
train_pad_lm_rag3 = make_dataset_pad_compact(
    df_pad_train,
    use_rag=True,
    n_max=None,
    k=3
)

val_pad_lm_rag3 = make_dataset_pad_compact(
    df_pad_val,
    use_rag=True,
    n_max=None,
    k=3
)

test_pad_lm_rag3 = make_dataset_pad_compact(
    df_pad_test,
    use_rag=True,
    n_max=None,
    k=3
)

print("Train:", len(train_pad_lm_rag3))
print("Val  :", len(val_pad_lm_rag3))
print("Test :", len(test_pad_lm_rag3))

print("\nExample prompt:\n")
print(train_pad_lm_rag3[0]["prompt"][:2000])

print("\nCompletion:\n")
print(train_pad_lm_rag3[0]["completion"])


# In[88]:


with open(pad_llm_dir / "train_pad_lm_rag3_compact.json", "w", encoding="utf-8") as f:
    json.dump(train_pad_lm_rag3, f, ensure_ascii=False, indent=2)

with open(pad_llm_dir / "val_pad_lm_rag3_compact.json", "w", encoding="utf-8") as f:
    json.dump(val_pad_lm_rag3, f, ensure_ascii=False, indent=2)

with open(pad_llm_dir / "test_pad_lm_rag3_compact.json", "w", encoding="utf-8") as f:
    json.dump(test_pad_lm_rag3, f, ensure_ascii=False, indent=2)


# In[89]:


train_pad_ds_rag3 = Dataset.from_list(train_pad_lm_rag3)
val_pad_ds_rag3 = Dataset.from_list(val_pad_lm_rag3)
test_pad_ds_rag3 = Dataset.from_list(test_pad_lm_rag3)


# In[90]:


sample = train_pad_lm_rag3[0]
print("Premier sample du train_pad_lm_rag3")
print(f"Label: {sample['label']}")
print(f"file_num: {sample['file_num']}")
print(f"\nRetrieved labels: {sample['retrieved_labels']}")
print(f"Retrieved scores: {sample['retrieved_scores']}")
print(f"\nMax retrieved score: {max(sample['retrieved_scores'])}")
print(f"\n Prompt (first 500 chars) ")
print(sample['prompt'][:500])


# ## Chat Template Formatting for Paderborn LoRA Training

# In[91]:


base_model_name = "meta-llama/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

print("Tokenizer loaded.")
print("Pad token:", tokenizer.pad_token)


# In[92]:


def format_training_example(example):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_PAD
        },
        {
            "role": "user",
            "content": example["prompt"]
        },
        {
            "role": "assistant",
            "content": example["completion"]
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )


def add_text_field(example):
    example["text"] = format_training_example(example)
    return example


train_pad_ds_rag3_fmt = train_pad_ds_rag3.map(add_text_field)
val_pad_ds_rag3_fmt = val_pad_ds_rag3.map(add_text_field)


# In[93]:


train_pad_ds_rag3_text = train_pad_ds_rag3_fmt.remove_columns(
    [c for c in train_pad_ds_rag3_fmt.column_names if c != "text"]
)

val_pad_ds_rag3_text = val_pad_ds_rag3_fmt.remove_columns(
    [c for c in val_pad_ds_rag3_fmt.column_names if c != "text"]
)


# In[94]:


from transformers import DataCollatorForLanguageModeling

ASSISTANT_HEADER_IDS = [128006, 78191, 128007, 271]


class CompletionOnlyCollator(DataCollatorForLanguageModeling):
    """Loss calculated ONLY on the assistant's response.
    All tokens before and including the assistant header are masked at -100.
    """
    def __init__(self, tokenizer, response_ids):
        super().__init__(tokenizer=tokenizer, mlm=False)
        self.response_ids = list(response_ids)

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        labels = batch["labels"].clone()
        tpl = self.response_ids
        n_tpl = len(tpl)

        for i in range(labels.size(0)):
            seq = batch["input_ids"][i].tolist()
            start = None

            for j in range(len(seq) - n_tpl + 1):
                if seq[j:j + n_tpl] == tpl:
                    start = j + n_tpl
                    break

            if start is not None:
                labels[i, :start] = -100
            else:
                labels[i, :] = -100

        batch["labels"] = labels
        return batch


collator_pad = CompletionOnlyCollator(
    tokenizer=tokenizer,
    response_ids=ASSISTANT_HEADER_IDS,
)


# In[95]:


_sample = train_pad_ds_rag3_fmt[0]

_enc = tokenizer(
    _sample["text"],
    return_tensors="pt",
    truncation=True,
    max_length=1536,
    padding="max_length"
)

_batch = collator_pad([
    {
        "input_ids": _enc["input_ids"][0].tolist(),
        "attention_mask": _enc["attention_mask"][0].tolist()
    }
])

active_tokens = (_batch["labels"][0] != -100).sum().item()
total_tokens = _batch["labels"][0].numel()

print("Active tokens:", active_tokens)
print("Total tokens:", total_tokens)
print("Active percentage:", 100 * active_tokens / total_tokens)

active_text = tokenizer.decode(
    _batch["input_ids"][0][_batch["labels"][0] != -100],
    skip_special_tokens=False
)

print("\nStart of active text:")
print(active_text[:500])


# ## Base Model and LoRA Adapter for Paderborn Fine-Tuning

# In[96]:


# os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# gc.collect()
# torch.cuda.empty_cache()

# base_model_name = "meta-llama/Llama-3.2-3B-Instruct"

# pad_lora_dir = output_dir / "llama32_lora_paderborn"
# pad_lora_dir.mkdir(parents=True, exist_ok=True)

# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True,
#     torch_dtype=torch.float16,
#     attn_implementation="eager"
# )

# base_model.config.use_cache = False
# base_model = prepare_model_for_kbit_training(base_model)

# lora_config_pad = LoraConfig(
#     r=16,
#     lora_alpha=32,
#     target_modules=[
#         "q_proj", "k_proj", "v_proj", "o_proj",
#         "gate_proj", "up_proj", "down_proj"
#     ],
#     lora_dropout=0.05,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# llm_model_pad = get_peft_model(base_model, lora_config_pad)

# llm_model_pad.print_trainable_parameters()


#  ## LoRA Trainer

# In[97]:


# training_args_pad = SFTConfig(
#     output_dir=str(output_dir / "llama32_lora_paderborn"),
#     num_train_epochs=5,
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=2,
#     eval_accumulation_steps=4,
#     gradient_accumulation_steps=4,
#     learning_rate=2e-4,
#     warmup_steps=10,
#     lr_scheduler_type="cosine",
#     logging_steps=20,
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     load_best_model_at_end=True,
#     metric_for_best_model="eval_loss",
#     greater_is_better=False,
#     fp16=False,
#    bf16=False,
#     gradient_checkpointing=True,
#     optim="paged_adamw_8bit",
#     report_to="none",
#     dataset_text_field="text",
#     max_length=1536,
#     packing=False,
#     seed=42,
#     max_grad_norm=1.0,
#     dataloader_num_workers=2,
# )
# trainer_pad = SFTTrainer(
#     model=llm_model_pad,
#     train_dataset=train_pad_ds_rag3_text,
#     eval_dataset=val_pad_ds_rag3_text,
#     args=training_args_pad,
#     data_collator=collator_pad,
#     processing_class=tokenizer
# )


# In[98]:


# gc.collect()
# torch.cuda.empty_cache()
# base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
# pad_lora_dir = output_dir / "llama32_lora_paderborn"


# trainer_pad.train()

# trainer_pad.save_model(str(pad_lora_dir))
# tokenizer.save_pretrained(str(pad_lora_dir))

# print("Adaptateur LoRA Paderborn sauvegardé dans :", str(pad_lora_dir))

# # del trainer_pad
# # del llm_model_pad
# # del base_model

# gc.collect()

# torch.cuda.empty_cache()


# ## Paderborn LoRA Trainer for evaluation

# In[99]:


# base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
# pad_lora_dir = output_dir / "llama32_lora_paderborn"


# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
# )

# tokenizer = AutoTokenizer.from_pretrained(
#     base_model_name,
#     trust_remote_code=True
# )

# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# tokenizer.padding_side = "right"

# base_model = AutoModelForCausalLM.from_pretrained(
#     base_model_name,
#     quantization_config=bnb_config,
#     device_map="auto",
#     trust_remote_code=True,
#     torch_dtype=torch.float16,
#     attn_implementation="eager"
# )

# model_pad = PeftModel.from_pretrained(
#     base_model,
#     pad_lora_dir
# )

# model_pad.eval()


# In[100]:


pad_lora_dir = output_dir / "llama32_lora_paderborn"

print("pad_lora_dir:", pad_lora_dir)
print("Exists:", pad_lora_dir.exists())
print("Is directory:", pad_lora_dir.is_dir())

if pad_lora_dir.exists():
    print("\nFiles inside:")
    for f in pad_lora_dir.iterdir():
        print("-", f.name)


# ## Paderborn LoRA Trainer inference

# In[101]:


def format_llama_prompt_pad(prompt):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_PAD
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def generate_diagnosis_paderborn(prompt, max_new_tokens=160):
    formatted_prompt = format_llama_prompt_pad(prompt)

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1536
    ).to(model_pad.device)

    with torch.no_grad():
        outputs = model_pad.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return generated.strip()


def extract_fault_class(text):
    m = re.search(
        r"Fault class\s*:\s*(Normal|Inner Race|Ball|Outer Race)",
        text,
        re.IGNORECASE
    )

    if not m:
        return "Unknown"

    label = m.group(1).lower()

    if label == "normal":
        return "Normal"
    if label == "inner race":
        return "Inner Race"
    if label == "outer race":
        return "Outer Race"
    if label == "ball":
        return "Ball"

    return "Unknown"


# In[ ]:


# pad_lora_results = []

# for i, sample in enumerate(tqdm(test_pad_lm_rag3, desc="Paderborn LoRA inference")):
#     pred_text = generate_diagnosis_paderborn(sample["prompt"])
#     pred_label = extract_fault_class(pred_text)

#     pad_lora_results.append({
#         "true_label": sample["label"],
#         "pred_label": pred_label,
#         "generated": pred_text,
#         "prompt": sample["prompt"]
#     })

#     if (i + 1) % 50 == 0:
#         with open(pad_llm_dir / "paderborn_lora_results_partial.json", "w", encoding="utf-8") as f:
#             json.dump(pad_lora_results, f, ensure_ascii=False, indent=2)


# y_true = [r["true_label"] for r in pad_lora_results]
# y_pred = [r["pred_label"] for r in pad_lora_results]

# pad_lora_metrics = {
#     "Experiment": "Paderborn LoRA + RAG compact k=3",
#     "Accuracy": accuracy_score(y_true, y_pred) * 100,
#     "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
#     "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0) * 100,
#     "F1-score": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
# }

# with open(pad_llm_dir / "paderborn_lora_results.json", "w", encoding="utf-8") as f:
#     json.dump(pad_lora_results, f, ensure_ascii=False, indent=2)

# with open(pad_llm_dir / "paderborn_lora_metrics.json", "w", encoding="utf-8") as f:
#     json.dump(pad_lora_metrics, f, ensure_ascii=False, indent=2)



# In[ ]:



# In[ ]:


results_path = pad_llm_dir / "paderborn_lora_results.json"

with open(results_path, "r", encoding="utf-8") as f:
    pad_results = json.load(f)

try:
    print(f"Length: {len(pad_lora_results)}")

    print("Distribution true_labels (mémoire):")
    print(Counter([r["true_label"] for r in pad_lora_results]))

    print("Distribution pred_labels (mémoire):")
    print(Counter([r["pred_label"] for r in pad_lora_results]))

    print(f"\nFichier == mémoire : {saved_results == pad_lora_results}")

except NameError:
    print("La variable pad_lora_results n'existe pas encore en mémoire.")
    pad_lora_results = pad_results
    print("pad_lora_results a été initialisée avec le contenu du fichier.")


y_true = [r["true_label"] for r in pad_results]
y_pred = [r["pred_label"] for r in pad_results]

print("\n Metrics recomputed from the saved file ")
print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")
print(f"F1 macro: {f1_score(y_true, y_pred, average='macro', zero_division=0) * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_true, y_pred, zero_division=0))


# In[ ]:


pad_true = np.array([label_map[r['true_label']] for r in pad_results])
pad_pred = np.array([label_map[r['pred_label']] for r in pad_results])

llm_results = {
    'model':     'LLaMA 3.2 3B + LoRA + RAG',
    'accuracy':  round(accuracy_score(pad_true, pad_pred) * 100, 2),
    'precision': round(precision_score(pad_true, pad_pred, average='macro', zero_division=0) * 100, 2),
    'recall':    round(recall_score(pad_true,    pad_pred, average='macro', zero_division=0) * 100, 2),
    'f1_score':  round(f1_score(pad_true,        pad_pred, average='macro', zero_division=0) * 100, 2),
}
print('LLM Results:', llm_results)
plot_cm(pad_true, pad_pred, 'LLaMA 3.2 LoRA', fig_dir / 'llm_cm.png')


# # Readability

# In[ ]:


def extract_explanation_text(generated: str) -> str:
    m_exp = re.search(r'Explanation:\s*(.+?)(?=\nRecommendation:|$)', generated, re.DOTALL)
    m_rec = re.search(r'Recommendation:\s*(.+?)$', generated, re.DOTALL)
    parts = []
    if m_exp: parts.append(m_exp.group(1).strip())
    if m_rec: parts.append(m_rec.group(1).strip())
    return ' '.join(parts)


texts_by_class = {'Normal': [], 'Inner Race': [], 'Outer Race': [], 'Ball': []}

for r in pad_results:
    text = extract_explanation_text(r['generated'])
    if len(text.split()) < 5:
        continue
    cls = r.get('pred_label', 'Unknown')

    if cls in texts_by_class:
        texts_by_class[cls].append(text)


def compute_readability(texts: dict, group_name: str) -> pd.DataFrame:
    rows = []
    for key, txts in texts.items():
        if not txts:
            continue
        agg = ' '.join(txts)
        if len(agg.split()) < 100:
            print(f"  {key}: moins de 100 mots ({len(agg.split())}), scores omis")
            continue
        rows.append({
            group_name:    key,
            'N_diag':      len(txts),
            'Words':       len(agg.split()),
            'Flesch RE':   round(textstat.flesch_reading_ease(agg), 2),
            'FK Grade':    round(textstat.flesch_kincaid_grade(agg), 2),
            'Gunning Fog': round(textstat.gunning_fog(agg), 2),
            'SMOG':        round(textstat.smog_index(agg), 2),
            'Dale-Chall':  round(textstat.dale_chall_readability_score(agg), 2),
        })
    return pd.DataFrame(rows)

print(" Readability by CLASS")
df_read_cls = compute_readability(texts_by_class, 'Class')
print(df_read_cls.to_string(index=False))


# ## CWRU LoRA for Zero-Shot Transfer to Paderborn

# In[ ]:


base_model_name = "meta-llama/Llama-3.2-3B-Instruct"
ckpt_dir = output_dir / "llama32_lora_revision_seed42"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    ckpt_dir,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.float16,
    attn_implementation="eager"
)

model = PeftModel.from_pretrained(
    base_model,
    ckpt_dir
)

model.eval()

print("CWRU LoRA model loaded successfully.")





def format_llama_prompt(prompt):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_PAD
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def generate_diagnosis_pad(prompt, max_new_tokens=160):
    formatted_prompt = format_llama_prompt(prompt)

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=3072
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )

    return generated.strip()


def extract_fault_class(text):
    m = re.search(
        r"Fault class\s*:\s*(Normal|Inner Race|Ball|Outer Race)",
        text,
        re.IGNORECASE
    )

    if not m:
        return "Unknown"

    label = m.group(1).lower()

    if label == "normal":
        return "Normal"
    if label == "inner race":
        return "Inner Race"
    if label == "outer race":
        return "Outer Race"
    if label == "ball":
        return "Ball"

    return "Unknown"





cwru_to_pad_results = []

for i, sample in enumerate(tqdm(test_pad_ds_rag3, desc="CWRU LoRA -> Paderborn inference")):
    pred_text = generate_diagnosis_pad(sample["prompt"])
    pred_label = extract_fault_class(pred_text)

    cwru_to_pad_results.append({
        "true_label": sample["label"],
        "pred_label": pred_label,
        "generated": pred_text
    })

    if (i + 1) % 50 == 0:
        with open(pad_llm_dir / "cwru_lora_on_paderborn_partial.json", "w", encoding="utf-8") as f:
            json.dump(cwru_to_pad_results, f, ensure_ascii=False, indent=2)
with open(pad_llm_dir / "cwru_lora_on_paderborn_results.json", "w", encoding="utf-8") as f:
    json.dump(cwru_to_pad_results, f, ensure_ascii=False, indent=2)





with open(pad_llm_dir / "cwru_lora_on_paderborn_results.json", "r", encoding="utf-8") as f:
  cwru_to_pad_results = json.load(f)





y_true = [r["true_label"] for r in cwru_to_pad_results]
y_pred = [r["pred_label"] for r in cwru_to_pad_results]

cwru_to_pad_metrics = {
    "Experiment": "CWRU LoRA -> Paderborn",
    "Accuracy": accuracy_score(y_true, y_pred) * 100,
    "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0) * 100,
    "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0) * 100,
    "F1-score": f1_score(y_true, y_pred, average="macro", zero_division=0) * 100,
}
with open(pad_llm_dir / "cwru_lora_on_paderborn_metrics.json", "w", encoding="utf-8") as f:
    json.dump(cwru_to_pad_metrics , f, ensure_ascii=False, indent=2)


# In[ ]:


# with open(pad_llm_dir / "cwru_lora_on_paderborn_metrics.json", "r", encoding="utf-8") as f:
#     cwru_to_pad_metrics= json.load(f)

# cwru_to_pad_metrics


# In[ ]:


# cwru_pad_true = np.array([label_map[r['true_label']] for r in cwru_to_pad_results])
# cwru_pad_pred = np.array([label_map[r['pred_label']] for r in cwru_to_pad_results])

# llm_results = {
#     'model':     'LLaMA 3.2 3B + LoRA + RAG',
#     'accuracy':  round(accuracy_score(cwru_pad_true, cwru_pad_pred) * 100, 2),
#     'precision': round(precision_score(cwru_pad_true, cwru_pad_pred, average='macro', zero_division=0) * 100, 2),
#     'recall':    round(recall_score(cwru_pad_true,    cwru_pad_pred, average='macro', zero_division=0) * 100, 2),
#     'f1_score':  round(f1_score(cwru_pad_true,        cwru_pad_pred, average='macro', zero_division=0) * 100, 2),
# }
# print('LLM Results:', llm_results)
# plot_cm(pad_true, pad_pred, 'LLaMA 3.2 LoRA', fig_dir / 'llm_cm.png')


# # Major-Revision Validation Suite


import subprocess, sys, shutil

PROJECT_DIR = Path('/home/yehoyakim/2026')   

def free_gpu_globals(names=('llm_model_pad', 'model_pad', 'base_model', 'model_inf',
                            'base_obj', 'exp2_base', 'model_exp2', 'trainer_pad')):
    g = globals()
    for n in names:
        g.pop(n, None)
    gc.collect(); torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        print(f'cuda:{i} alloué={torch.cuda.memory_allocated(i)/2**30:.2f} GiB | '
              f'réservé={torch.cuda.memory_reserved(i)/2**30:.2f} GiB')

def _gpu_free_gib():
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    res = {}
    for line in out.strip().splitlines():
        uuid, free_mib = [x.strip() for x in line.split(',')]
        res[uuid] = int(free_mib) / 1024
    return res

def pick_gpu(min_gib=10.0):
    """UUID du GPU (parmi ceux autorisés) avec le plus de mémoire libre."""
    allowed = []
    for i in range(torch.cuda.device_count()):
        u = str(torch.cuda.get_device_properties(i).uuid)
        allowed.append(u if u.startswith('GPU-') else 'GPU-' + u)
    free = _gpu_free_gib()
    cand = {u: free[u] for u in allowed if u in free}
    if not cand:
        raise RuntimeError('Impossible de faire correspondre les GPU visibles avec nvidia-smi.')
    print('[GPU] libre (GiB):', {f'cuda:{i}': round(cand.get(u, float('nan')), 1)
                                 for i, u in enumerate(allowed)})
    uuid, gib = max(cand.items(), key=lambda kv: kv[1])
    if gib < min_gib:
        raise RuntimeError(f'Aucun GPU autorisé avec >= {min_gib} GiB libres '
                           f'(meilleur : {gib:.1f} GiB). Vérifiez nvidia-smi.')
    return uuid

def train_isolated(train_ds, val_ds, out_dir, seed, epochs=5, min_gib=10.0):
    out_dir = Path(out_dir)
    done = Path(str(out_dir) + '_training_metrics.json')
    if done.exists() and (out_dir / 'adapter_config.json').exists():
        print(f'[skip] {out_dir.name} déjà entraîné')
        return json.loads(done.read_text())

    data_dir = Path(str(out_dir) + '_data')
    shutil.rmtree(data_dir, ignore_errors=True)
    train_ds.save_to_disk(str(data_dir / 'train'))
    val_ds.save_to_disk(str(data_dir / 'val'))

    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': pick_gpu(min_gib),
           'TOKENIZERS_PARALLELISM': 'false'}
    cmd = [sys.executable, str(PROJECT_DIR / 'train_worker.py'),
           '--data', str(data_dir), '--out', str(out_dir),
           '--seed', str(seed), '--epochs', str(epochs)]
    subprocess.run(cmd, env=env, cwd=str(PROJECT_DIR), check=True)
    shutil.rmtree(data_dir, ignore_errors=True)
    return json.loads(done.read_text())

def load_exp2_model(adapter_dir=None, device=0):
    adapter_dir = str(adapter_dir or EXP2_ADAPTER_DIR)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_id, device_map={'': device}, quantization_config=bnb_config,
        low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation='eager')
    m = PeftModel.from_pretrained(base, adapter_dir); m.eval()
    t = AutoTokenizer.from_pretrained(adapter_dir); t.pad_token = t.eos_token
    return m, t

# ## A. Multi-seed LoRA robustness (5 independent training runs)
# 

# In[ ]:
free_gpu_globals() 

REVISION_SEEDS = [11, 22, 33, 44, 55]
multiseed_rows = []
all_seed_predictions = {}

def infer_all(mdl, tok, samples, cond, seed):
    out = []
    for sample in tqdm(samples, desc=f'seed={seed} {cond}'):
        gen = llm_predict_with(mdl, tok, sample['prompt'])
        out.append({'seed': seed, 'condition': cond, 'true_label': sample['label'],
                    'pred_label': parse_class(gen), 'file_num': int(sample['file_num']),
                    'generated': gen, 'retrieved_labels': sample['retrieved_labels']})
    return out

for seed in REVISION_SEEDS:
    adapter_dir = output_dir / f'llama32_lora_exp2_lowload_seed{seed}'
    pred_file = output_dir / f'exp2_multiseed_predictions_seed{seed}.json'

    if pred_file.exists():                       # reprise après crash
        saved = json.loads(pred_file.read_text())
        rr, rn = saved['rag'], saved['norag']
        print(f'[reprise] seed {seed}: prédictions déjà calculées')
    else:
        # 1) entraînement dans un sous-processus
        train_isolated(train_exp2_ds, val_exp2_ds, adapter_dir, seed=seed, epochs=5)
        # 2) inférence dans ce processus
        gc.collect(); torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(
            base_model_id, device_map={'': 0}, quantization_config=bnb_config,
            low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation='eager')
        mdl = PeftModel.from_pretrained(base, str(adapter_dir)); mdl.eval()
        tok = AutoTokenizer.from_pretrained(str(adapter_dir)); tok.pad_token = tok.eos_token

        rr = infer_all(mdl, tok, test_exp2_rag, 'RAG_k3', seed)
        rn = infer_all(mdl, tok, test_exp2_no_rag, 'No_RAG', seed)
        pred_file.write_text(json.dumps({'rag': rr, 'norag': rn}, indent=2))
        del mdl, base; gc.collect(); torch.cuda.empty_cache()

    all_seed_predictions[seed] = {'rag': rr, 'norag': rn}
    for cond, res in [('RAG_k3', rr), ('No_RAG', rn)]:
        yt = [x['true_label'] for x in res]; yp = [x['pred_label'] for x in res]
        multiseed_rows.append({'seed': seed, 'condition': cond,
            'accuracy': accuracy_score(yt, yp),
            'macro_f1': f1_score(yt, yp, labels=label_names_order, average='macro', zero_division=0),
            'parse_rate': np.mean([p != 'Unknown' for p in yp])})


multi_df=pd.DataFrame(multiseed_rows)
multi_df.to_csv(output_dir/'exp2_multiseed_summary.csv',index=False)
display(multi_df)
print('\nMean ± SD by condition:')
display(multi_df.groupby('condition')[['accuracy','macro_f1','parse_rate']].agg(['mean','std']))


from scipy.stats import ttest_rel, wilcoxon
wide=multi_df.pivot(index='seed',columns='condition',values='macro_f1')
delta=wide['RAG_k3']-wide['No_RAG']
print('Per-seed ΔF1:',delta.to_dict())
print('Paired t-test:',ttest_rel(wide['RAG_k3'],wide['No_RAG']))
print('Wilcoxon:',wilcoxon(wide['RAG_k3'],wide['No_RAG'],alternative='two-sided'))

stats_multiseed={
    'rag_mean':float(wide['RAG_k3'].mean()),'rag_sd':float(wide['RAG_k3'].std(ddof=1)),
    'norag_mean':float(wide['No_RAG'].mean()),'norag_sd':float(wide['No_RAG'].std(ddof=1)),
    'delta_mean':float(delta.mean()),'delta_sd':float(delta.std(ddof=1)),
    'paired_t_p':float(ttest_rel(wide['RAG_k3'],wide['No_RAG']).pvalue),
    'wilcoxon_p':float(wilcoxon(wide['RAG_k3'],wide['No_RAG']).pvalue),
}
with open(output_dir/'exp2_multiseed_statistical_tests.json','w') as f: json.dump(stats_multiseed,f,indent=2)


# ## B. Retrieval sensitivity: k, Hit@1/Hit@3, prompt length, and latency
# 

# In[ ]:


import time

model_exp2, tok_exp2 = load_exp2_model()

# def retrieval_quality_exp2(k_values=(1,3,5,7)):
#     rows=[]
#     for _,row in tqdm(df_test_exp2.iterrows(),total=len(df_test_exp2),desc='Retrieval quality'):
#         cand=retrieve_exp2(row['text'],k=max(k_values)+12)
#         cand=[r for r in cand if r['file_num']!=row['file_num']]
#         rec={'file_num':row['file_num'],'true_label':row['label'],'query_load':row['load'],'defect_size':row['defect_size']}
#         for k in k_values:
#             top=cand[:k]
#             rec[f'hit@{k}']=int(any(r['label']==row['label'] for r in top))
#             rec[f'same_load@{k}']=np.mean([r['load']==row['load'] for r in top]) if top else np.nan
#             if pd.notna(row['defect_size']):
#                 rec[f'same_severity_proxy@{k}']=np.mean([r['defect_size']==row['defect_size'] for r in top]) if top else np.nan
#         rows.append(rec)
#     return pd.DataFrame(rows)

# retrieval_q=retrieval_quality_exp2((1,3,5,7))
# retrieval_q.to_csv(output_dir/'exp2_retrieval_quality_full.csv',index=False)
# print(retrieval_q[[c for c in retrieval_q if c.startswith('hit@')]].mean())


# k_rows=[]
# for k in [0,1,3,5,7]:
#     ds=make_dataset_exp2(df_test_exp2,use_rag=(k>0),k=k)
#     preds=[]; prompt_tokens=[]; lat=[]
#     for sample in tqdm(ds,desc=f'k={k}'):
#         prompt_tokens.append(len(tok_exp2(sample['prompt'],add_special_tokens=False)['input_ids']))
#         t0=time.perf_counter(); gen=llm_predict_with(model_exp2,tok_exp2,sample['prompt']); lat.append(time.perf_counter()-t0)
#         preds.append(parse_class(gen))
#     yt=[s['label'] for s in ds]
#     k_rows.append({'k':k,'macro_f1':f1_score(yt,preds,labels=label_names_order,average='macro',zero_division=0),
#                    'accuracy':accuracy_score(yt,preds),'mean_prompt_tokens':np.mean(prompt_tokens),
#                    'mean_latency_s':np.mean(lat),'parse_rate':np.mean([p!='Unknown' for p in preds])})

# k_df=pd.DataFrame(k_rows); k_df.to_csv(output_dir/'exp2_k_sensitivity.csv',index=False); display(k_df)


# ## C. Embedding-model sensitivity
# 

# In[ ]:


gc.collect()
torch.cuda.empty_cache()

# EMBEDDING_MODELS=[
#     'all-MiniLM-L6-v2',
#     'all-mpnet-base-v2',
#     'multi-qa-mpnet-base-dot-v1',
# ]
# embedding_rows=[]


# csv_path = output_dir / 'exp2_embedding_sensitivity.csv'

# if csv_path.exists():
#     embedding_df = pd.read_csv(csv_path)
#     embedding_rows = embedding_df.to_dict('records')
#     done_models = set(embedding_df['embedding_model'].tolist())
#     print(f"[REPRISE] Modèles déjà calculés : {done_models}")
# else:
#     embedding_rows = []
#     done_models = set()

# for emb_name in EMBEDDING_MODELS:

#     if emb_name in done_models:
#         print(f"[SKIP] {emb_name} déjà évalué, passage au suivant.")
#         continue

#     device = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
#     emb_model = SentenceTransformer(emb_name, device=device)
#     corpus = df_tr2['text'].tolist()
#     E = emb_model.encode(corpus, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True).astype('float32')
#     idx = faiss.IndexFlatIP(E.shape[1]); idx.add(E)

#     def _retrieve(qtext, k=3):
#         q = emb_model.encode([qtext], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
#         scores, ids = idx.search(q, min(k, idx.ntotal))
#         return [{'text': df_tr2.iloc[int(i)]['text'], 'label': df_tr2.iloc[int(i)]['label'],
#                  'diagnosis': df_tr2.iloc[int(i)]['diagnosis'], 'file_num': df_tr2.iloc[int(i)]['file_num'],
#                  'load': df_tr2.iloc[int(i)]['load'], 'defect_size': df_tr2.iloc[int(i)]['defect_size'], 'score': float(s)}
#                 for s, i in zip(scores[0], ids[0])]

#     hit1 = []; hit3 = []
#     for _, row in tqdm(df_test_exp2.iterrows(), total=len(df_test_exp2), desc=f"Retrieval {emb_name}"):
#         cand = [r for r in _retrieve(row['text'], k=11) if r['file_num'] != row['file_num']]
#         hit1.append(any(r['label'] == row['label'] for r in cand[:1]))
#         hit3.append(any(r['label'] == row['label'] for r in cand[:3]))

#     test_emb = make_dataset_exp2(df_test_exp2, use_rag=True, k=3, retrieve_fn=_retrieve)
#     preds = []
#     for sample in tqdm(test_emb, desc=f'LLM {emb_name}'):
#         with torch.no_grad():
#             gen = llm_predict_with(model_exp2, tok_exp2, sample['prompt'])
#         preds.append(parse_class(gen))
        
#     yt = [s['label'] for s in test_emb]
    
#     # Ajouter le résultat du modèle courant
#     embedding_rows.append({
#         'embedding_model': emb_name,
#         'Hit@1': np.mean(hit1),
#         'Hit@3': np.mean(hit3),
#         'macro_f1': f1_score(yt, preds, labels=label_names_order, average='macro', zero_division=0),
#         'accuracy': accuracy_score(yt, preds)
#     })
    

#     pd.DataFrame(embedding_rows).to_csv(csv_path, index=False)
    
#     del emb_model, E, idx
#     gc.collect()
#     torch.cuda.empty_cache()


csv_path = output_dir / 'exp2_embedding_sensitivity.csv'
if csv_path.exists():
    embedding_df = pd.read_csv(csv_path)
    display(embedding_df)
else:
    print("Le fichier CSV d'embeddings n'existe pas encore.")

free_gpu_globals()

# ## D. Component ablations: operating metadata and expected fault frequencies
# 

# In[ ]:


# def run_encoding_ablation(name, include_metadata, include_freqs, seed=42):
#     # Fresh copies avoid contaminating the full representation.
#     tr=df_tr2.copy(); va=df_va2.copy(); te=df_test_exp2.copy()
#     for d in [tr,va,te]:
#         d['text']=d.apply(lambda r: encode_to_text_variant(r,include_metadata,include_freqs),axis=1)
#         d['diagnosis']=d.apply(make_output,axis=1)

#     emb_model=SentenceTransformer('all-MiniLM-L6-v2',device='cuda:1' if torch.cuda.device_count()>1 else 'cuda:0')
#     E=emb_model.encode(tr.text.tolist(),convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True).astype('float32')
#     idx=faiss.IndexFlatIP(E.shape[1]); idx.add(E)
#     def retr(qtext,k=3):
#         q=emb_model.encode([qtext],convert_to_numpy=True,normalize_embeddings=True).astype('float32')
#         scores,ids=idx.search(q,min(k,idx.ntotal))
#         return [{'text':tr.iloc[int(i)].text,'label':tr.iloc[int(i)].label,'diagnosis':tr.iloc[int(i)].diagnosis,
#                  'file_num':tr.iloc[int(i)].file_num,'load':tr.iloc[int(i)].load,'defect_size':tr.iloc[int(i)].defect_size,'score':float(s)}
#                 for s,i in zip(scores[0],ids[0])]

#     tr_s=make_dataset_exp2(tr,True,3,retr); va_s=make_dataset_exp2(va,True,3,retr); te_s=make_dataset_exp2(te,True,3,retr)
#     tr_ds=samples_to_training_dataset(tr_s); va_ds=samples_to_training_dataset(va_s)
#     outdir=output_dir/f'ablation_{name}_seed{seed}'
#     train_isolated(tr_ds, va_ds, outdir, seed=seed, epochs=5)

#     base=AutoModelForCausalLM.from_pretrained(base_model_id,device_map={'':0},quantization_config=bnb_config,
#         low_cpu_mem_usage=True,trust_remote_code=True,attn_implementation='eager')
#     mdl=PeftModel.from_pretrained(base,str(outdir)); mdl.eval(); tok=AutoTokenizer.from_pretrained(str(outdir)); tok.pad_token=tok.eos_token
#     preds=[]
#     for sample in tqdm(te_s,desc=name): preds.append(parse_class(llm_predict_with(mdl,tok,sample['prompt'])))
#     yt=[s['label'] for s in te_s]
#     result={'ablation':name,'include_metadata':include_metadata,'include_expected_freqs':include_freqs,
#             'macro_f1':f1_score(yt,preds,labels=label_names_order,average='macro',zero_division=0),
#             'accuracy':accuracy_score(yt,preds),'parse_rate':np.mean([p!='Unknown' for p in preds])}
#     del mdl,base,emb_model; gc.collect(); torch.cuda.empty_cache()
#     return result

# ablation_results = []

# ablation_results.append({
#     'ablation': 'full',
#     'include_metadata': True,
#     'include_expected_freqs': True,
#     'macro_f1': res_exp2_rag['f1_score'] / 100,
#     'accuracy': res_exp2_rag['accuracy'] / 100,
#     'parse_rate': res_exp2_rag['parse_rate'] / 100
# })

# name = 'no_operating_metadata'
# outdir = output_dir / f'ablation_{name}_seed42'

# tr_no_meta = df_tr2.copy(); va_no_meta = df_va2.copy(); te_no_meta = df_test_exp2.copy()
# for d in [tr_no_meta, va_no_meta, te_no_meta]:
#     d['text'] = d.apply(lambda r: encode_to_text_variant(r, False, True), axis=1)
#     d['diagnosis'] = d.apply(make_output, axis=1)

# emb_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0')
# E = emb_model.encode(tr_no_meta.text.tolist(), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False).astype('float32')
# idx = faiss.IndexFlatIP(E.shape[1]); idx.add(E)

# def retr_nometa(qtext, k=3):
#     q = emb_model.encode([qtext], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
#     scores, ids = idx.search(q, min(k, idx.ntotal))
#     return [{'text': tr_no_meta.iloc[int(i)].text, 'label': tr_no_meta.iloc[int(i)].label, 'diagnosis': tr_no_meta.iloc[int(i)].diagnosis,
#              'file_num': tr_no_meta.iloc[int(i)].file_num, 'load': tr_no_meta.iloc[int(i)].load, 'defect_size': tr_no_meta.iloc[int(i)].defect_size, 'score': float(s)}
#             for s, i in zip(scores[0], ids[0])]

# te_s_nometa = make_dataset_exp2(te_no_meta, True, 3, retr_nometa)


# base = AutoModelForCausalLM.from_pretrained(base_model_id, device_map={'': 0}, quantization_config=bnb_config, low_cpu_mem_usage=True, trust_remote_code=True, attn_implementation='eager')
# mdl = PeftModel.from_pretrained(base, str(outdir)); mdl.eval()
# tok = AutoTokenizer.from_pretrained(str(outdir)); tok.pad_token = tok.eos_token

# preds = [parse_class(llm_predict_with(mdl, tok, sample['prompt'])) for sample in tqdm(te_s_nometa, desc=name)]
# yt = [s['label'] for s in te_s_nometa]

# ablation_results.append({
#     'ablation': name, 'include_metadata': False, 'include_expected_freqs': True,
#     'macro_f1': f1_score(yt, preds, labels=label_names_order, average='macro', zero_division=0),
#     'accuracy': accuracy_score(yt, preds),
#     'parse_rate': np.mean([p != 'Unknown' for p in preds])
# })

# del mdl, base, emb_model; gc.collect(); torch.cuda.empty_cache()


# ablation_results.append(run_encoding_ablation('no_expected_fault_frequencies', True, False, seed=42))
# ablation_results.append(run_encoding_ablation('features_only', False, False, seed=42))

# ablation_df = pd.DataFrame(ablation_results)
# ablation_df.to_csv(output_dir / 'exp2_component_ablation.csv', index=False)
# display(ablation_df)

ablation_csv = output_dir / 'exp2_component_ablation.csv'
if ablation_csv.exists():
    ablation_df = pd.read_csv(ablation_csv)
    display(ablation_df)
else:
    print("Le fichier exp2_component_ablation.csv n'a pas encore été généré.")

# ## E. Factual/physical consistency and failure-case analysis
# 

# In[ ]:


# def parse_scientific_number_after_band(text, band):
#     # Accept forms such as "BPFI band energy (1.158e-07)".
#     m=re.search(rf'{band}[^\n]*?energy\s*\(?\s*([0-9.+\-eE]+)',text,re.I)
#     if not m: return np.nan
#     try: return float(m.group(1))
#     except: return np.nan

# def parse_expected_frequency(text):
#     m=re.search(r'expected frequency\s*([0-9.]+)\s*Hz',text,re.I)
#     return float(m.group(1)) if m else np.nan

# band_for={'Inner Race':'BPFI','Outer Race':'BPFO','Ball':'BSF'}
# energy_col={'Inner Race':'bpfi_energy','Outer Race':'bpfo_energy','Ball':'bsf_energy'}
# freq_key={'Inner Race':'f_BPFI','Outer Race':'f_BPFO','Ball':'f_BSF'}

# consistency=[]
# for row,res in zip(df_test_exp2.itertuples(index=False),results_rag):
#     pred=res['pred_label']; gen=res['generated']; exp=parse_explanation(gen)
#     item={'file_num':row.file_num,'true_label':row.label,'pred_label':pred,'generated':gen}
#     if pred in band_for:
#         band=band_for[pred]
#         item['mentions_expected_band']=int(band.lower() in exp.lower())
#         mentioned_e=parse_scientific_number_after_band(exp,band)
#         actual_e=float(getattr(row,energy_col[pred]))
#         item['mentioned_energy']=mentioned_e; item['actual_energy']=actual_e
#         item['energy_relative_error']=abs(mentioned_e-actual_e)/(abs(actual_e)+1e-15) if np.isfinite(mentioned_e) else np.nan
#         mentioned_f=parse_expected_frequency(exp)
#         expected_f=fault_frequencies(float(row.fr))[freq_key[pred]]
#         item['mentioned_frequency']=mentioned_f; item['expected_frequency']=expected_f
#         item['frequency_abs_error_hz']=abs(mentioned_f-expected_f) if np.isfinite(mentioned_f) else np.nan
#         # Conservative automatic consistency: correct physical band and, if numbers are stated, close to the query values.
#         item['auto_physically_consistent']=int(
#             item['mentions_expected_band']==1 and
#             (not np.isfinite(mentioned_e) or item['energy_relative_error']<=0.10) and
#             (not np.isfinite(mentioned_f) or item['frequency_abs_error_hz']<=3.0)
#         )
#     else:
#         item['mentions_expected_band']=np.nan; item['auto_physically_consistent']=np.nan
#     consistency.append(item)

# consistency_df=pd.DataFrame(consistency)
# consistency_df.to_csv(output_dir/'exp2_explanation_physical_consistency.csv',index=False)
# print('Automatic physical-consistency rate among fault predictions:',consistency_df.auto_physically_consistent.mean())

# # RAG transition categories on identical windows.
# fail=[]
# for i,(rr,rn,sample) in enumerate(zip(results_rag,results_no_rag,test_exp2_rag)):
#     t=rr['true_label']; cr=rr['pred_label']==t; cn=rn['pred_label']==t
#     if (not cn) and cr: cat='RAG fixes error'
#     elif cn and (not cr): cat='RAG introduces error'
#     elif not cr: cat='Both wrong'
#     else: cat='Both correct'
#     retrieved_wrong=bool(sample['retrieved_labels']) and not any(x==t for x in sample['retrieved_labels'])
#     fail.append({'idx':i,'file_num':rr['file_num'],'true_label':t,'rag_pred':rr['pred_label'],'norag_pred':rn['pred_label'],
#                  'category':cat,'retrieved_labels':'|'.join(sample['retrieved_labels']),'retrieval_miss':retrieved_wrong,
#                  'rag_generated':rr['generated'],'norag_generated':rn['generated']})
# fail_df=pd.DataFrame(fail)
# fail_df.to_csv(output_dir/'exp2_failure_case_catalog.csv',index=False)
# print(fail_df.category.value_counts())
# print('Retrieval misses:',fail_df.retrieval_miss.mean())

# # Expert-review sample: balanced across transition categories where possible.
# expert_parts=[]
# for cat,g in fail_df.groupby('category'):
#     expert_parts.append(g.sample(min(25,len(g)),random_state=42))
# expert_sample=pd.concat(expert_parts,ignore_index=True)
# expert_sample['expert_explanation_correct']=''
# expert_sample['expert_recommendation_appropriate']=''
# expert_sample['expert_notes']=''
# expert_sample.to_csv(output_dir/'expert_review_sample_100.csv',index=False)
# print('Expert-review form saved:', output_dir/'expert_review_sample_100.csv')


# # ## F. Dedicated deep time-series baseline: InceptionTime-lite on raw vibration
# # 

# # In[ ]:


# from torch.utils.data import Dataset as TorchDataset, DataLoader
# import torch.nn.functional as F

# class CWRURawWindowDataset(TorchDataset):
#     def __init__(self, records, allowed_files):
#         allowed=set(int(x) for x in allowed_files); self.records=records; self.items=[]
#         for ri,rec in enumerate(records):
#             if int(rec['file_num']) not in allowed: continue
#             n=(len(rec['signal'])-window_size)//window_step+1
#             for wi in range(max(0,n)):
#                 self.items.append((ri,wi*window_step))
#     def __len__(self): return len(self.items)
#     def __getitem__(self,idx):
#         ri,start=self.items[idx]; rec=self.records[ri]
#         x=rec['signal'][start:start+window_size].astype(np.float32).copy()
#         x=(x-x.mean())/(x.std()+1e-6)
#         return torch.from_numpy(x[None,:]), torch.tensor(label_map[rec['label']],dtype=torch.long)

# class InceptionModule1D(nn.Module):
#     def __init__(self,in_ch,out_ch=32,kernels=(9,19,39),bottleneck=32):
#         super().__init__(); b=min(bottleneck,in_ch) if in_ch>1 else 1
#         self.bottleneck=nn.Conv1d(in_ch,b,1,bias=False)
#         self.branches=nn.ModuleList([nn.Conv1d(b,out_ch,k,padding=k//2,bias=False) for k in kernels])
#         self.pool_branch=nn.Sequential(nn.MaxPool1d(3,stride=1,padding=1),nn.Conv1d(in_ch,out_ch,1,bias=False))
#         self.bn=nn.BatchNorm1d(out_ch*(len(kernels)+1))
#     def forward(self,x):
#         z=self.bottleneck(x); ys=[conv(z) for conv in self.branches]+[self.pool_branch(x)]
#         return F.relu(self.bn(torch.cat(ys,dim=1)))

# class InceptionTimeLite(nn.Module):
#     def __init__(self,n_classes=4):
#         super().__init__(); self.b1=InceptionModule1D(1,24); self.b2=InceptionModule1D(96,24); self.b3=InceptionModule1D(96,24)
#         self.head=nn.Linear(96,n_classes)
#     def forward(self,x):
#         x=self.b1(x); x=self.b2(x); x=self.b3(x); x=x.mean(dim=-1); return self.head(x)

# def train_timeseries_baseline(train_files_,val_files_,test_files_,seed=42,epochs=40,batch_size=32):
#     set_seed(seed); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     tr=CWRURawWindowDataset(records,train_files_); va=CWRURawWindowDataset(records,val_files_); te=CWRURawWindowDataset(records,test_files_)
#     trl=DataLoader(tr,batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
#     val=DataLoader(va,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
#     tel=DataLoader(te,batch_size=batch_size,shuffle=False,num_workers=4,pin_memory=True)
#     m=InceptionTimeLite().to(device); opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=1e-4)
#     best=-1; best_state=None; patience=8; bad=0
#     for ep in range(epochs):
#         m.train()
#         for xb,yb in trl:
#             xb=xb.to(device,non_blocking=True); yb=yb.to(device,non_blocking=True)
#             opt.zero_grad(); loss=F.cross_entropy(m(xb),yb); loss.backward(); opt.step()
#         m.eval(); yt=[]; yp=[]
#         with torch.no_grad():
#             for xb,yb in val:
#                 pred=m(xb.to(device)).argmax(1).cpu().numpy(); yp.extend(pred); yt.extend(yb.numpy())
#         vf=f1_score(yt,yp,average='macro')
#         print(f'epoch {ep+1:02d} val macro-F1={vf:.4f}')
#         if vf>best+1e-4:
#             best=vf; best_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}; bad=0
#         else:
#             bad+=1
#             if bad>=patience: break
#     m.load_state_dict(best_state); m.eval(); yt=[];yp=[]
#     with torch.no_grad():
#         for xb,yb in tel:
#             pred=m(xb.to(device)).argmax(1).cpu().numpy(); yp.extend(pred); yt.extend(yb.numpy())
#     return {'seed':seed,'accuracy':accuracy_score(yt,yp),'macro_f1':f1_score(yt,yp,average='macro'),
#             'report':classification_report(yt,yp,target_names=label_names,output_dict=True,zero_division=0),'y_true':yt,'y_pred':yp}


# high_files=sorted(df_test_exp2.file_num.unique())
# deep_rows=[]
# for seed in [11,22,33,44,55]:
#     r=train_timeseries_baseline(low_train_files,low_val_files,high_files,seed=seed)
#     deep_rows.append({'seed':seed,'accuracy':r['accuracy'],'macro_f1':r['macro_f1']})
#     if seed==42: pass

# deep_df=pd.DataFrame(deep_rows); deep_df.to_csv(output_dir/'exp2_inceptiontime_multiseed.csv',index=False)
# display(deep_df)
# print(deep_df.agg(['mean','std']))


# # ## G. Paderborn split audit and severity limitation
# # 

# # In[ ]:



# pad_split_audit=pd.DataFrame([
#     {'split':'train','n_files':df_pad_train.file_num.nunique(),'n_windows':len(df_pad_train)},
#     {'split':'validation','n_files':df_pad_val.file_num.nunique(),'n_windows':len(df_pad_val)},
#     {'split':'test','n_files':df_pad_test.file_num.nunique(),'n_windows':len(df_pad_test)},
# ])
# pad_split_audit['severity_validated']=False  # all Paderborn targets use a uniform Medium placeholder
# pad_split_audit.to_csv(pad_output_dir/'paderborn_split_audit_revision.csv',index=False)
# display(pad_split_audit)

# assert set(df_pad_train.file_num).isdisjoint(set(df_pad_val.file_num))
# assert set(df_pad_train.file_num).isdisjoint(set(df_pad_test.file_num))
# assert set(df_pad_val.file_num).isdisjoint(set(df_pad_test.file_num))
# print('Paderborn leakage check passed at recording-file level.')
# print('Severity is NOT evaluated on Paderborn because equivalent defect-size annotations are unavailable.')


# # ## H. Reproducibility manifest
# # 

# # In[ ]:


# import platform, sys
# manifest={
#     'python':sys.version,
#     'platform':platform.platform(),
#     'torch':torch.__version__,
#     'cuda':torch.version.cuda,
#     'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
#     'base_model':base_model_id,
#     'embedding_model_main':'all-MiniLM-L6-v2',
#     'generation':{
#         'do_sample':False,'max_new_tokens_cwru':150,'max_new_tokens_paderborn':160,
#         'repetition_penalty_cwru':1.1,'temperature':None,'top_p':None,'top_k':None,
#         'decoding':'greedy/deterministic'
#     },
#     'lora':{'r':16,'alpha':32,'dropout':0.05,'target_modules':['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']},
#     'revision_seeds':[11,22,33,44,55],
#     'cross_load_protocol':'LoRA + RAG corpus trained on 0-1 HP only; complete 2-3 HP test set',
# }
# with open(output_dir/'reproducibility_manifest_revision.json','w') as f: json.dump(manifest,f,indent=2)
# print(json.dumps(manifest,indent=2))






