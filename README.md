# MELORA

**MELORA: Medication Recommendation with Long and Short
Patient Health State Adaptively Fusion**

For reproduction of medication prediction results in our paper, see instructions below.

---

## Overview

This repository contains code necessary to run the **MELORA** model.

MELORA is an end-to-end medication recommendation framework that explicitly models patient temporal health states at multiple time scales. It integrates long-term stable health conditions and short-term transient clinical changes through a time–frequency decomposition and adaptive fusion mechanism.

Longitudinal patient history information and drug–drug interaction (DDI) knowledge are jointly leveraged to provide accurate, safe, and personalized medication combination recommendations.

MELORA is evaluated on real-world clinical datasets **MIMIC-III** and **MIMIC-IV**, where it consistently outperforms state-of-the-art medication recommendation models across multiple effectiveness metrics, while maintaining competitive or lower DDI rates.


## Requirements

- PyTorch >= 2.9.0  
- Python >= 3.10 

---

## Running the Code

### Data Preprocessing
1.Download [MIMIC-III](https://mimic.mit.edu/docs/gettingstarted/) and put DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv in ./data/mimic-iii

2.Download [MIMIC-IV](https://mimic.mit.edu/docs/iv/) and put DIAGNOSES_ICD.csv, PRESCRIPTIONS.csv, PROCEDURES_ICD.csv in ./data/mimic-iii

Run the following commands in sequence to preprocess the MIMIC-III and MIMIC-IV datasets:

```python
cd ./data/mimic-iii/
python processing_mimic-iii.py
python processing_mimic-iii-ATC4.py
```
```python
cd ./data/mimic-iv/
python processing_mimic-iv.py
python processing_mimic-iv-ATC4.py
```
### MELORA

```python
cd .
bash run_mimic_iii.sh
bash run_mimic_iv.sh

```
