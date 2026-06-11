# Interaction Quality Modelling in Spoken Dialogue Systems

Official Implementation for our Interspeech 2026 paper: **"A System-Agnostic Approach to Modelling Interaction Quality in Spoken Dialogue Systems"**

## About The Project
This project compares two approaches to training models to classify interaction quality: **system-dependent (SD)** and **system-agnostic** (SA). The system-dependent approach involves using system-log data, such as ASR confidence scores and Dialogue Manager state, whereas the system-agnostic approach involves using speech-based features derived from audio recordings of the interaction. To make this comparison, we train Long Short Term Memory (LSTM) models on features and interaction quality labels derived from the **CMU Let's Go (LEGO) corpus**, a publicly available corpus of spoken interactions between a user and a bus information system. For more information, please read our [Interspeech Paper]().

## Getting Started

Follow these steps to set up the environment and run the code locally.

### Prerequisites

* Python 3.10
* [Anaconda](https://www.anaconda.com/) or Miniconda
* (Optional) CUDA-compatible GPU for faster training

### Installation

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/yourusername/interaction-quality-modelling.git](https://github.com/yourusername/interaction-quality-modelling.git)
   cd interaction-quality-modelling

2. Create the virtual environment using the provided environment.yml file
   ```bash
   conda env create -f environment.yml
   conda activate model_iq_env

## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

## License

Distributed under the project_license. See `LICENSE.txt` for more information.

## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@email_client.com

## Acknowledgments

* []()
* []()
* []()
