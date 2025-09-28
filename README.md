# EEG_Seizure_Detection

This repository contains the code accompanying the master's thesis *"Anomaly Detection in EEG Data using Lifelong Learning"*

## Project structure
- Data preprocessing is implemented in the Jupyter notebook `preprocessing.ipynb`, which provides step-by-step control over the data preparation process

- Models are implemented in Python scripts (`.py` files), organized according to the chosen anomaly detection method and training strategy

### Implemented models

- Isolation Forest:
    - `IF_naive.py`
    - `IF_replay.py`

- Autoencoder
    - `ae_naive.py`
    - `ae_replay.py`

- One-Class SVM
    - `SVM_naive.py`
    - `SVM_replay.py`

## Dataset
The experiments are conducted on the [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)  
- The dataset contains EEG recordings from 23 pediatric patients with intractable seizures 
- Each recording is provided in EDF format, accompanied by annotations specifying seizure intervals
- In preprocessing, common EEG channels across recordings are selected, signals are filtered (1–45 Hz), and spectral features are extracted using Short-Time Fourier Transform (STFT)  
- Features are aggregated into frequency bands (delta, theta, alpha, beta, gamma) and standardized before model training  

## Results
The models are evaluated under lifelong learning scenarios with both **Naive** and **Replay-based** strategies.  
The following aspects are measured:
- **ROC-AUC** – seizure detection accuracy for each patient (concept)  
- **Continual Average** – aggregated performance over the sequence of concepts  
- **Backward Transfer (BWT)** – how learning new concepts affects performance on previous ones  
- **Forward Transfer (FWT)** – how prior knowledge helps with new concepts 
- **Computation Time** – runtime evaluation for each training scenario 
- **Memory Usage** – monitored with a dedicated callback to assess resource efficiency

Results are automatically saved in `.json` format for further analysis and visualization



