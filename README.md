# Interaction Quality Modelling in Spoken Dialogue Systems

Official Implementation for our Interspeech 2026 paper: **"A System-Agnostic Approach to Modelling Interaction Quality in Spoken Dialogue Systems"**

## About The Project
This project compares two approaches to training models to **classify interaction quality**:
* **System-dependent (SD):** Uses system-log data, such as ASR confidence scores and dialogue manager states.
* **System-agnostic (SA):** Uses speech-based features derived from audio recordings of interactions. 

To compare these approaches, we train **Long Short Term Memory (LSTM)** models on features and interaction quality labels derived from the **CMU Let's Go (LEGO) corpus**.

For further details, please refer to our [Interspeech Paper](https://doi.org/10.21437/Interspeech.2026-1152) and the LEGO corpus paper by Schmitt et al. (2012).

## Setup & Installation

### Prerequisites
* Python 3.10
* [Anaconda](https://www.anaconda.com/) or Miniconda
* (Optional) CUDA-compatible GPU for faster training

### Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/prgering/interaction-quality-modelling-sds.git
   cd interaction-quality-modelling-sds
   ```

2. Create and activate the virtual environment:
   ```bash
   conda env create -f environment.yml
   conda activate iq_predict_env
   ```

### Data Preparation
Download the LEGO corpus from the [University of Bamberg Resources Website](https://www.uni-bamberg.de/ds/ressourcen/lego/).

Unzip the downloaded folder and place it in the `data/raw/` directory.

## Running the Static Feature Pipeline

### Step A: System Log Preprocessing

Cleans raw system logs, expands semantic parses, and extracts prompt/utterance text embeddings using pretrained self-supervised models.

```bash
python scripts/preprocess_system_logs.py
```

### Step B: User Speech Transcription

Filters user speech using a two-pass Silero VAD, applies Gaussian noise masking, and transcribes audio via WhisperX.

```bash
python scripts/transcribe_user_speech.py
```

### Step C: System Speech Alignment

Extract system transcript and timestamps from mixed transcript using WhisperX and fuzzy alignment.

```bash
python scripts/align_system_prompts.py
```

### Step D: Speech Feature Extraction

Combine user and agent transcripts to extract exchange-level speech features.

> Note: Run on HPC via Slurm**

```bash
# Extracts OpenSMILE features only
sbatch slurm/scripts/extract_speech_features.sh static_speech

# Extracts full speech features (requires GPU)
sbatch slurm/scripts/extract_speech_features.sh speech
```

### Step E: Filter System Log Features

Filter the System-Derived (SD) features to exclude dialogues with missing audio files or no user or agent speech. 

> Prerequisite: Run only after validated user and agent transcripts are generated.

```bash
python scripts/filter_system_features.py
```

### Step F: Prepare Feature Sets for Machine Learning

Prepares features for modelling by applying scaling, PCA, and dataset splitting.

```bash
python scripts/prepare_features_for_modelling.py
```

### Step G: Hyperparameter Tuning with Static Features

Performs 10-fold cross-validation grid search for interaction quality classifiers with LSTM architectures.

> Slurm Configuration: Hyperparameter sweeps are driven by slurm array jobs. Generate parameter configuration text files with `scripts/generate_hyperparam_configs.py`. Ensure `#SBATCH --array` size matches the total configuration count.

```bash
# Speech features
sbatch slurm/scripts/hyperparam_tune_speech_lstm.sh

# System features
sbatch slurm/scripts/hyperparam_tune_system_lstm.sh
```

### Step H: Analyse Hyperparameter Tuning Results

Analyses cross-validation performance across hyperparameter sweeps to determine optimal configurations for each feature set.

```bash
python scripts/analyse_cv_results.py
```

### Step I: Final Model Evaluation
Trains models on the full training set and evaluates performance on the held-out test set.

> Prerequisite: place the best hyperparameter configurations from **Step H** in `slurm/configs/best_static_model_params.txt` (one space-separated configuration per line).

```bash
sbatch slurm/scripts/train_eval_best_models_static.sh
```

### Step J: Analyse Evaluation Results
Performs pairwise permutation tests with Holm-Bonferroni correction to compare model performance (Macro-F1 & Recall) across model variants.

> Note: Run this script only after completing **both** the static and Fine-Tuned Pipeline evaluation steps.

```bash
python scripts/analyse_eval_results.py
```

---
## Running the Fine-Tuned Pipeline
> **Note on Terminology:** In this repository, the terms **Fine-Tuned Pipeline** and **End-to-End (E2E)** refer to the same architecture. Scripts and configuration files related to this pipeline use the `end2end` naming convention.
> Prerequisite: Complete Steps A &mdash; E from the Static Feature Pipeline first. 
> For Step D, run the `static_speech` mode to extract only OpenSMILE features.

### Step F: Hyperparameter Tuning
Trains fine-tuned pipeline on training set and evaluates on the validation set to determine optimal window-size, encoder learning rate, and frozen encoder layers. 

> Note: Uses the optimal hyperparameters previously tuned in the Static Pipeline.
> 1. Ensure best static parameters are defined in `slurm/scripts/hyperparam_tune_end2end.sh`
> 2. Generate configuration files with `scripts/generate_hyperparam_configs.py` for fine-tuning parameters.

```bash
# Speech feature tuning
MODE=speech sbatch hyperparam_tune_end2end.sh

# System feature tuning
MODE=system sbatch hyperparam_tune_end2end.sh
```

### Step G: Analysing Fine-Tuning Results
Analyses validation set metrics across the fine-tuning sweeps to select the top-performing architectures.

```bash
python scripts/analyse_cv_results_end2end.py
```

### Step H: Final Model Evaluation
Trains models on the full training set and evaluates performance on the held-out test set.

> Prerequisite: place the best hyperparameter configurations from **Step G** in `slurm/configs/best_end2end_params.txt` (one space-separated configuration per line).

```bash
sbatch slurm/scripts/train_eval_best_models_end2end.sh
```

> Final Step: Proceed to **Step J** under the Static Pipeline section (`python scripts/analyse_eval_results.py`) to run joint statistical significance testing across both static and fine-tuned models.

## License

Distributed under the . See `LICENSE.txt` for more information.


## Citation
If you use this repistory or build upon this work, please cite our paper:

```bibtex
@inproceedings{gering26_interspeech,
  title     = {{A System-Agnostic Approach to Modelling Interaction Quality in Spoken Dialogue Systems}},
  author    = {Paul Gering and Roger K. Moore},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {3400--3404},
  doi       = {10.21437/Interspeech.2026-1152},
  issn      = {2958-1796},
}
```


## References
```bibtex
@inproceedings{bain2022whisperx,
	title        = {{WhisperX}: Time-Accurate Speech Transcription of Long-Form Audio},
	author       = {Bain, Max and Huh, Jaesung and Han, Tengda and Zisserman, Andrew},
	year         = 2023,
	booktitle    = {Interspeech 2023},
	volume       = {2023-},
	pages        = {4489--4493},
}
@misc{SileroVAD,
	title        = {{Silero VAD}: pre-trained enterprise-grade Voice Activity Detector (VAD), Number Detector and Language Classifier},
	author       = {{Silero Team}},
	year         = 2024,
	journal      = {GitHub repository},
	howpublished = {\url{https://github.com/snakers4/silero-vad}},
}
@inproceedings{schmitt2012parameterized,
	title        = {A Parameterized and Annotated Spoken Dialog Corpus of the {CMU Let's Go Bus Information System}},
	author       = {Schmitt, Alexander  and Ultes, Stefan  and Minker, Wolfgang},
	year         = 2012,
	booktitle    = {Proceedings of the Eighth International Conference on Language Resources and Evaluation ({LREC}'12)},
	pages        = {3369--3373},
}
@inproceedings{eyben2010opensmile,
	title        = {{openSMILE} -- The Munich Versatile and Fast Open-Source Audio Feature Extractor},
	author       = {Eyben, Florian and W{\"o}llmer, Martin and Schuller, Bj{\"o}rn},
	year         = 2010,
	booktitle    = {MM'10 - Proceedings of the ACM Multimedia 2010 International Conference},
	pages        = {1459--1462},
	doi          = {10.1145/1873951.1874246},
}
```