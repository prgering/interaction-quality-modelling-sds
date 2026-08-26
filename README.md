# Interaction Quality Modelling in Spoken Dialogue Systems

Official Implementation for our Interspeech 2026 paper: **"A System-Agnostic Approach to Modelling Interaction Quality in Spoken Dialogue Systems"**

## About The Project
This project compares two approaches to training models to **classify interaction quality**: **system-dependent (SD)** and **system-agnostic (SA)**. The system-dependent approach involves using system-log data, such as ASR confidence scores and dialogue manager states, whereas the system-agnostic approach involves using speech-based features derived from audio recordings of the interaction. To make this comparison, we train Long Short Term Memory (LSTM) models on features and interaction quality labels derived from the **CMU Let's Go (LEGO) corpus**, a publicly available corpus of spoken interactions between a user and a bus information system. For more information, please read our [Interspeech Paper]() and the LEGO corpus paper by Schmitt et al. (2012).

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

2. Create the virtual environment using the provided environment.yml file
   ```bash
   conda env create -f environment.yml
   conda activate model_iq_env

## Usage

### 1. Data Preparation

You must first download the LEGO corpus from the [University of Bamberg Resources Website](https://www.uni-bamberg.de/ds/ressourcen/lego/).

Once you have unzipped the downloaded folder, place it in the 'data/raw/' directory of this repository.

### 2. Running the Pipeline

#### Step A: System Log Preprocessing

Cleans raw system logs, expands semantic parses, and extracts prompt/utterance text embeddings using pretrained self-supervised models.
```bash
python scripts/preprocess_system_logs.py
```

#### Step B: User Speech Transcription

Filters user speech using a two-pass Silero VAD, applies Gaussian noise masking, and transcribes audio via WhisperX.

```bash
python scripts/transcribe_user_speech.py
```

#### Step C: System Speech Alignment

Extract system transcript and timestamps from mixed transcript using WhisperX and fuzzy alignment.

```bash
python scripts/align_system_prompts.py
```

#### Step D: Speech Feature Extraction

> **Note:** User and agent transcripts should be manually verified before running this step.

Combine user and agent transcripts and extract exchange-level speech features (OpenSMILE acoustic features, pretrained speech embeddings, and text embeddings).

**1. Local Execution (CPU / Fast Mode)**

Extracts OpenSMILE features only

```bash
python -m scripts/extract_speech_features --skip-embeddings
```

Extracts full speech features for small subset of data

```bash
python -m scripts/extract_speech_features --debug
```

**2. HPC Execution via Slurm**

Extracts OpenSMILE features only

```bash
sbatch slurm/scripts/extract_speech_features.sh static_speech
```

Extracts full speech features (Requires GPU)

```bash
sbatch slurm/scripts/extract_speech_features.sh speech
```

#### Step E: Filter System Log Features

#### Step F: Prepare Feature Sets for Machine Learning

#### Step G: Hyperparameter Tuning with LSTM

## License

Distributed under the project_license. See `LICENSE.txt` for more information.

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
```