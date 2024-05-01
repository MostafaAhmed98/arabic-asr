# Arabic Speech Recognition with NeMo

This project demonstrates Arabic speech recognition using NVIDIA's NeMo toolkit. It provides a simple interface for transcribing Arabic speech from uploaded or recorded audio files.

## Requirements

- Python 3.10+
- NeMo 1.23.0
- Librosa
- Soundfile
- Gradio

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MostafaAhmed98/arabic-asr.git
   ```

2. Install the required dependencies:

   ```bash
   ## Install NeMo
    BRANCH = 'r1.23.0'
    python -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[all]
   ## Install Librosa, soundfile and gradio
    pip install gradio
    pip install soundfile
    pip install librosa
   ```

## Usage

1. Run the script `arabic_asr.py`:

   ```bash
   python app.py
   ```

2. Once the script is running, a Gradio interface will launch in your browser.

3. Upload or record an Arabic audio file for transcription.

4. Wait for the transcription to be completed. The predicted text will be displayed on the interface.

## Description

- `arabic_asr.py`: This script contains the main functionality for Arabic speech recognition. It defines functions for converting audio files to the required format (16kHz) and for transcribing the audio using a pre-trained NeMo model.

- `convert_wav_to_16k`: This function converts the input audio file to 16kHz sampling rate and mono type if it's not already in that format.

- `loading_nemo_and_prediction`: This function loads the pre-trained NeMo model for Arabic speech recognition and performs transcription on the processed audio file.

- `predict`: This function acts as a bridge between the Gradio interface and the NeMo model. It takes an uploaded audio file, converts it to the required format, and then performs transcription.

- `demo`: This part of the script creates a Gradio interface using the `gr.Interface` class, defining the input and output components.

## Credits

- This project utilizes the NeMo toolkit developed by NVIDIA.
- Gradio is used for creating the user interface.

---
