"""TRIBE v2 demo script extracted from notebooks/tribe_demo.ipynb."""

# %% [markdown]
# # TRIBE v2 Demo: Predicting Brain Responses to Naturalistic Stimuli
#
# [TRIBE v2](https://github.com/facebookresearch/tribev2) is a deep multimodal brain encoding model that predicts **fMRI brain responses** to naturalistic stimuli â€” video, audio, and text.
#
# It combines state-of-the-art feature extractors â€” **LLaMA 3.2** (text), **V-JEPA2** (video), and **Wav2Vec-BERT** (audio) â€” into a unified Transformer that maps multimodal representations onto the cortical surface (**fsaverage5**, ~20k vertices).
#
# In this notebook, we will:
# 1. Load a pretrained TRIBE v2 model from HuggingFace
# 2. Predict brain responses to a **video** clip
# 3. Predict brain responses to **audio** generated from text
# 4. Visualize the predicted activity on a 3D brain surface

# %% [markdown]
# ## Setup (for Colab users)
#
# 1. Activate the GPU (Menu > Runtime > Change runtime)
# 2. Run the command below
# 3. Restart your environment for the new packages to be taken into account

# %%
# Colab setup cell converted to a local setup reminder:
#   pip install -r requirements.txt

# %% [markdown]
# ## Loading the model
#
# We load TRIBE v2 model from [HuggingFace Hub](https://huggingface.co/facebook/tribev2). On the first run, this downloads the model checkpoint and config (~1 GB). Subsequent runs use the cached version.
#
# We also initialize a `PlotBrain` object for 3D brain surface visualization using the **fsaverage5** mesh.

# %%
from tribev2.demo_utils import TribeModel, download_file
from tribev2.plotting import PlotBrain
from pathlib import Path

CACHE_FOLDER = Path("./cache")

model = TribeModel.from_pretrained(
    "facebook/tribev2",
    cache_folder=CACHE_FOLDER,
)
plotter = PlotBrain(mesh="fsaverage5")

# %% [markdown]
# ## Predict brain responses to a video
#
# Given a video file, TRIBE v2 automatically:
# 1. **Extracts audio** from the video track
# 2. **Transcribes speech** into word-level events with timestamps using [**WhisperX**](https://github.com/m-bain/whisperx)
# 3. **Extracts visual features** (DINOv2 + V-JEPA2) and **audio features** (Wav2Vec-BERT) and **text features** (LLaMA 3.2)
# 4. **Predicts fMRI activity** at each time step (1 TR = 1 second) across the cortical surface
#
# Below, we download a sample video ([Sintel trailer](https://durian.blender.org/)), build an events dataframe, and run the model.

# %%
video_path = CACHE_FOLDER / "sample_video.mp4"
url = "https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4"
download_file(url, video_path)
df = model.get_events_dataframe(video_path=video_path)
display(df.head(8)[["type", "start", "duration", "filepath", "text", "context"]])

# %% [markdown]
# ### Run the model
#
# We feed the events dataframe to `model.predict()`, which extracts features for each modality, runs them through the Transformer, and returns predicted brain activity.
#
# NOTE: you will have to request access to the Llama-3.2 model using your HuggingFace account.
#
# The output `preds` has shape `(n_timesteps, n_vertices)` â€” one prediction per second of stimulus, with ~20k cortical vertices. The `segments` list contains the corresponding time segments with their associated events.

# %%
preds, segments = model.predict(events=df)
print(f"Predictions shape: {preds.shape}  (n_timesteps, n_vertices)")

# %% [markdown]
# ### Visualize predictions on the brain surface
#
# We plot the predicted fMRI activity for the first 15 time steps on the fsaverage5 cortical mesh. Each panel shows one second of predicted activity, with the corresponding stimulus frame displayed below. Predictions are offset by 5 seconds in the past, in order to compensate for the hemodynamic lag.
#
# We see that as the image appears on the screen, the visual cortex lights up (t=4s), followed by the language network when the character starts to speak (t=12s).

# %%
n_timesteps = 15
fig = plotter.plot_timesteps(preds[:n_timesteps], segments=segments[:n_timesteps], cmap="fire", norm_percentile=99, vmin=.6, alpha_cmap=(0, .2), show_stimuli=True)

# %% [markdown]
# ## Predict brain responses to text (via text-to-speech)
#
# TRIBE v2 can also predict brain responses to **text** input. Since the model was trained on naturalistic audio/video stimuli, text is first converted to speech using Google Text-to-Speech (gTTS), then transcribed back to obtain precise word-level timings.
#
# Below, we use a passage from Shakespeare's *Hamlet* as input.

# %%
text = """
To be or not to be, that is the question.
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep,
No more; and by a sleep to say we end
The heartache and the thousand natural shocks
"""

text_path = CACHE_FOLDER / "shakespeare.txt"
text_path.write_text(text)

df = model.get_events_dataframe(text_path=text_path)
display(df.head(8)[["type", "start", "duration", "filepath", "text", "context"]])

# %% [markdown]
# ### Run the model
#
# Same as before â€” we pass the events dataframe to `model.predict()` to get brain activity predictions for each time step.

# %%
preds, segments = model.predict(events=df)
print(f"Predictions shape: {preds.shape}  (n_timesteps, n_vertices)")

# %% [markdown]
# ### Visualize predictions on the brain surface
#
# Again, we visualize the first 15 seconds of predicted activity. For audio-only stimuli, the stimulus display shows the spoken words at each time step.

# %%
n_timesteps = 15
fig = plotter.plot_timesteps(preds[:n_timesteps], segments=segments[:n_timesteps], cmap="fire", norm_percentile=99, vmin=.6, alpha_cmap=(0, .2), show_stimuli=True)

# %%

