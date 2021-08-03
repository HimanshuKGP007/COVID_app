import streamlit as st
#from synthesizer.inference import Synthesizer
#from encoder import inference as encoder
#from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import os
import librosa
import glob
from helper import draw_embed, create_spectrogram, read_audio, record, save_record, preprocess, get_dataframe, scaler_transform
import pickle
import pandas as pd

#from setup_logging import setup_logging
# setup_logging()
# logger = logging.getLogger('app')

"# Streamlit showcase"
model_load_state = st.text("Loading pretrained models...")

# seed = 42
# low_mem = False
# num_generated = 0
# enc_model_fpath = Path("encoder/saved_models/pretrained.pt")
# syn_model_dir = Path("synthesizer/saved_models/logs-pretrained/")
# voc_model_fpath = Path("vocoder/saved_models/pretrained/pretrained.pt")
# encoder.load_model(enc_model_fpath)
# synthesizer = Synthesizer(
#     syn_model_dir.joinpath("taco_pretrained"), low_mem=low_mem, seed=seed
# )
# vocoder.load_model(voc_model_fpath)

loaded_model = pickle.load(open(r'C:\Users\DELL\COVID_app\model\finalized_model.sav', 'rb'))

model_load_state.text("Loaded pretrained models!")

st.header("1. Record your own voice")

filename = st.text_input("Choose a filename: ")

if st.button(f"Click to Record"):
    if filename == "":
        st.warning("Choose a Username.")
    else:
        record_state = st.text("Recording...")
        duration = 5  # seconds
        fs = 22050
        myrecording = record(duration, fs)
        record_state.text(f"Saving sample as {filename}.wav")

        path_myrecording = f"./samples/{filename}.wav"

        save_record(path_myrecording, myrecording, fs)
        record_state.text(f"Done! Saved sample as {filename}.wav")

        st.audio(read_audio(path_myrecording))

        fig = create_spectrogram(path_myrecording)
        st.pyplot(fig)

if st.button(f'Classify'):
        cnn = loaded_model
        path_myrecording = f"./samples/{filename}.wav"
        with st.spinner("Classifying the chord"):
            retro = preprocess(path_myrecording)
            retro1 = get_dataframe(retro)
            retro2 = scaler_transform(retro1)

            chord = cnn.predict(retro2)
        st.success("Classification completed")
        #st.header()
        # st.write("### The recorded chord is **", chord + "**")
        # if chord == 'N/A':
        #     st.write("Please record sound first")
        # st.write("\n")

