# TRIBE v2 Demo (Local Copy)

This workspace contains a local copy of the public Colab demo notebook from:

https://colab.research.google.com/github/facebookresearch/tribev2/blob/main/tribe_demo.ipynb

## What This Is

TRIBE v2 is a multimodal brain-encoding model that predicts fMRI cortical responses from naturalistic stimuli:

- Video
- Audio
- Text (via text-to-speech)

The demo does four things:

1. Loads pretrained model weights from Hugging Face (`facebook/tribev2`)
2. Runs prediction on a sample video
3. Runs prediction on Shakespeare text after converting text to speech
4. Visualizes predicted activity on the `fsaverage5` cortical surface

## Project Structure

- `notebooks/tribe_demo.ipynb`: exact notebook copy from the upstream GitHub repo
- `scripts/tribe_demo.py`: notebook cells exported into a Python script (cell markers preserved)
- `requirements.txt`: local dependency install target

## Quick Start

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Open and run:

   - Notebook workflow: `notebooks/tribe_demo.ipynb`
   - Script/cell workflow: `scripts/tribe_demo.py`

## Notes

- First model load downloads about 1 GB of checkpoints/config.
- Some extractors (for example Llama-based text features) may require Hugging Face access approval and credentials.
- GPU is strongly recommended.
