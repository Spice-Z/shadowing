# Speech Shadowing Practice

A Python application for practicing speech shadowing - a language learning technique where learners repeat speech immediately after hearing it. This tool helps users improve their pronunciation by comparing their speech with a target audio recording.

## Features

- Audio file format conversion to WAV
- Detailed audio comparison using Dynamic Time Warping (DTW)
- Analysis of multiple speech aspects:
  - Pronunciation accuracy
  - Speaking speed
  - Pitch patterns
  - Volume/energy patterns
  - Overall clarity
- Detailed feedback on areas for improvement
- Numerical scoring system

## Requirements

- Python 3.7+
- Required Python packages:
  - soundfile
  - numpy
  - librosa
  - scipy
  - fastdtw
  - pydub
  - matplotlib

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/speech-shadowing.git
   cd speech-shadowing
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use: env\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your target audio file as `target_audio.wav` in the project directory
2. Record or place your shadowing attempt as `user_audio.wav`
3. Run the program:
   ```bash
   python speech_shadowing.py
   ```

The program will analyze both audio files and provide:
- An overall similarity score
- Specific feedback on areas for improvement
- Detailed analysis of pronunciation, timing, and intonation

## Output

The program provides:
- A numerical score (0-100) indicating overall similarity
- Specific feedback on:
  - Speaking speed differences
  - Pitch pattern mismatches
  - Volume/energy differences
  - Pronunciation clarity issues

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 