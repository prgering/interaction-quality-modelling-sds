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

Cleans raw system logs, expands semantic parses, and extracts prompt/utterance text embeddings.
```bash
python scripts/preprocess_system_logs.py
```

#### Step B: User Speech Transcription

Filters user speech using a two-pass Silero VAD, applies Gaussian noise masking, and transcribes audio via Whisper.

```bash
python scripts/transcribe_user_speech.py
```

#### Step C: System Speech Alignment

Extract system transcript and timestamps from mixed transcript using ASR and fuzzy alignment.

```bash
python scripts/align_system_prompts.py
```

#### Step D: Speech Feature Extraction

#### Step E: Filter System Log Features

#### Step F: Prepare Feature Sets for Machine Learning

#### Step G: Hyperparameter Tuning with LSTM

## License

Distributed under the project_license. See `LICENSE.txt` for more information.
