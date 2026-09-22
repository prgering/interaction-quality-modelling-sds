# Interaction Quality Modelling in Spoken Dialogue Systems

Official Implementation for our Interspeech 2026 paper: **"A System-Agnostic Approach to Modelling Interaction Quality in Spoken Dialogue Systems"**

## About The Project
This project compares two approaches to training models to **classify interaction quality**: **system-dependent (SD)** and **system-agnostic (SA)**. The system-dependent approach involves using system-log data, such as ASR confidence scores and dialogue manager states, whereas the system-agnostic approach involves using speech-based features derived from audio recordings of the interaction. To make this comparison, we train Long Short Term Memory (LSTM) models on features and interaction quality labels derived from the **CMU Let's Go (LEGO) corpus**, a publicly available corpus of spoken interactions between a user and a bus information system. For more information, please read our [Interspeech Paper](https://doi.org/10.21437/Interspeech.2026-1152) and the LEGO corpus paper by Schmitt et al. (2012).

## Getting Started

Follow these steps to set up the environment and run the code locally.

### Prerequisites

* Python 3.10
* [Anaconda](https://www.anaconda.com/) or Miniconda
* (Optional) CUDA-compatible GPU for faster training

### Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/interaction-quality-modelling.git
   cd interaction-quality-modelling
   ```

2. Create the virtual environment using the provided environment.yml file
   ```bash
   conda env create -f environment.yml
   conda activate model_iq_env
   ```

### Data Preparation
You must download the LEGO corpus from the [University of Bamberg Resources Website](https://www.uni-bamberg.de/ds/ressourcen/lego/).

Once you have unzipped the downloaded folder, place it in the 'data/raw/' directory of this repository.

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

Combine user and agent transcripts and extract exchange-level speech features (OpenSMILE acoustic features, pretrained speech embeddings, and text embeddings).

**1. Local Execution (CPU / Fast Mode)**

* a. Extracts OpenSMILE features only

```bash
python -m scripts/extract_speech_features --skip-embeddings
```

* b. Extracts full speech features for small subset of data

```bash
python -m scripts/extract_speech_features --debug
```

**2. HPC Execution via Slurm**

* a. Extracts OpenSMILE features only

```bash
sbatch slurm/scripts/extract_speech_features.sh static_speech
```

* b. Extracts full speech features (Requires GPU)

```bash
sbatch slurm/scripts/extract_speech_features.sh speech
```

### Step E: Filter System Log Features

Excluding dialogues from the system-derived features if there are missing audio files or dialogues with no user or agent speech. This step can only be run once validated user and agent transcripts have been produced.

```bash
python scripts/filter_system_features.py
```

### Step F: Prepare Feature Sets for Machine Learning

Prepares features for modelling by loading, preprocessing, and splitting the data. PCA and scaling are applied to embedding features, whereas non-embedding features are
only scaled.

```bash
python scripts/prepare_features_for_modelling.py
```

### Step G: Hyperparameter Tuning with Static Features

Performs hyperparameter tuning on LSTM classifier with static features (10-fold grouped cross-validation grid search).

Hyperparameter sweeps are driven by slurm array jobs. Generate parameter configurations using `scripts/generate_hyperparam_configs.py`, and ensure the `#SBATCH --array` size matches the total number of lines in the generated text file.

* a. Speech features

```bash
sbatch slurm/scripts/hyperparam_tune_speech_lstm.sh
```

* b. System features

```bash
sbatch slurm/scripts/hyperparam_tune_system_lstm.sh
```

### Step H: Analyse Hyperparameter Tuning Results

Analyses hyperparameter sweep results to identify optimal parameter configurations for both feature sets.

```bash
python scripts/analyse_cv_results.py
```

### Step I: Final Model Evaluation
Trains models on the full training set using the best-performing hyperparameters and evaluates predictions on the test set. 

Before running, place the best-performing hyperparameter configuration for each feature set (from Step H) in `slurm/configs/best_model_params.txt` (formatted as one configuration per line, with each parameter space-separated and the parameter order matching the order expected in the shell script).

```bash
sbatch slurm/scripts/train_eval_best_models.sh
```

### Step J: Analyse Evaluation Results
Performs multiple permutation tests with Holm-Bonferroni correction to compare Macro-F1 scores of different model variants on the test set.

```bash
python scripts/analyse_eval_results.py
```

Note: Ensure you have completed both the static feature pipeline and the fine-tuned pipeline steps before running this analysis as it evaluates predictions across all static and fine-tuned model variants simultaneously.

## Running the Fine-Tuned Pipeline
Complete Steps A &mdash; E from the Frozen Pipeline Instructions. 

For Step D (Speech Feature Extraction), only extract the OpenSMILE features by specifying the `static_speech` mode.

### Step F:

### Step G:


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