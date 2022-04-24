import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from dataset import DemixingAudioDataset
from unet_model import UNet
from utils import calculate_chunk_size, load_model_and_optimizer
from temp_files.utils import plot_specgram, plot_waveform
from constants import *
import copy
import torchaudio
import os

INSTRUMENTS = ("bass", "drums", "vocals", "other")
KERNEL_SIZE = 4
SAMPLING_RATE = 44100
CLIP_TIME = 15
SAMPLE_BLOCK_DEPTH = 1
BOTTLENECK_DEPTH = 1
EXPERIMENTS =  ["b6_lr0.0001_d0.2_scl0.5/epoch_2-2022_4_23_4_54",
                "b6_lr0.0001_d0.2_scl0.5/epoch_10-2022_4_23_7_51",
                "b6_lr0.0001_d0.2_scl0.5/epoch_50-2022_4_23_7_51",
                "b1_lr0.01_dN_scl0.5/epoch_1-2022_4_23_13_57"]

import streamlit as st

st.title("50.039: Theory and Practice of Deep Learning - Audio Demixing Project")
st.subheader("Input audio (.wav)")
user_input = {}

# Input user audio wav 
user_input["input_audio"] = st.file_uploader(
    "Pick an audio to test"
)




# Remove warning for pyplot
st.set_option('deprecation.showPyplotGlobalUse', False)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


if user_input["input_audio"]:

    # Create sound player for orignal audio
    st.write("Original audio input")
    st.audio(user_input["input_audio"])

    selection = st.selectbox("Select model to run",EXPERIMENTS,0)
    
    # load model configurations
    model_path = "checkpoints/{}".format(selection)
    audio_model, optimizer, loaded_data = load_model_and_optimizer(device,model_path)
    chunk_size = calculate_chunk_size(CLIP_TIME*SAMPLING_RATE,loaded_data['block_count'])
    
    # Prepare input data: Convert audio to tensor 
    input_waveform, _ = torchaudio.load(user_input["input_audio"])
    # Prepare input data: Perform clipping and chunking
    input_waveform_chunks = torch.split(input_waveform,chunk_size,dim=-1)

    show_waveform = st.checkbox('Show Waveform')
    sample_length = 0
    if show_waveform:
        sample_length = st.slider(label='length of sample', min_value=0, max_value=10, key=4, value=5)
        sample = input_waveform[:,SAMPLING_RATE*4:SAMPLING_RATE*(sample_length+4)]
        torchaudio.save(os.path.join("tempDir","input.wav"), sample, SAMPLING_RATE)
        st.pyplot(plot_waveform(sample,SAMPLING_RATE,title="Original {} secs Waveform".format(sample_length)))
    show_spectogram = st.checkbox('Show Spectogram')
    if show_spectogram:
        if sample_length: 
            st.pyplot(plot_specgram(sample,SAMPLING_RATE,title="Original {} secs Spectogram".format(sample_length)))
        else:
            sample_length = st.slider(label='length of sample', min_value=0, max_value=10, key=4, value=5)
            sample = input_waveform[:,SAMPLING_RATE*4:SAMPLING_RATE*(sample_length+4)]
            torchaudio.save(os.path.join("tempDir","input.wav"), sample, SAMPLING_RATE)
            st.pyplot(plot_specgram(sample,SAMPLING_RATE,title="Original {} secs Spectogram".format(sample_length)))
    if show_waveform or show_spectogram:
        st.write("Original {} second sample".format(sample_length))
        st.audio(os.path.join("tempDir","input.wav"))

    loading = st.empty()
    loading.write("loading....")

    output = {}

    audio_model.eval()

    # Disable gradient tracking
    with torch.no_grad():

        # Keep list of  predicted component chunks
        pred_chunk_list_dict = {i:list() for i in INSTRUMENTS}
    
        # Iterate through each chunk
        for segment_idx in range(len(input_waveform_chunks)):

            # Skip chunks without proper size
            input_data = input_waveform_chunks[segment_idx]
            if (input_data.shape[-1]<chunk_size): continue

            # Move chunk to GPU or alternative
            input_data = input_data.to(device)
            input_data = torch.reshape(input_data,(1,2,-1))

            for instr in INSTRUMENTS:

                # Save pred
                pred = audio_model(input_data,instr)
                pred_chunk_list_dict[instr].append(pred.cpu().detach())
        for i in INSTRUMENTS:
            output_audio = torch.cat(pred_chunk_list_dict[i],dim=-1).reshape((2,-1))
            torchaudio.save(os.path.join("tempDir","{}.wav".format(i)), output_audio, SAMPLING_RATE,format="wav")
            st.write("Demixed {} output".format(i))
            st.audio(os.path.join("tempDir","{}.wav".format(i)))
            show_waveform = st.checkbox('Show Demixed {} Waveform'.format(i))
            if show_waveform:
                sample = output_audio[:,SAMPLING_RATE*4:SAMPLING_RATE*(sample_length+4)]
                torchaudio.save(os.path.join("tempDir","input.wav"), sample, SAMPLING_RATE)
                st.pyplot(plot_waveform(sample,SAMPLING_RATE,title="Demixed {} {} secs Waveform".format(i,sample_length)))
            show_spectogram = st.checkbox('Show Demixed {} Spectogram'.format(i))
            if show_spectogram:
                if sample_length: 
                    st.pyplot(plot_specgram(sample,SAMPLING_RATE,title="Demixed {} {} secs Spectogram".format(i,sample_length)))
                else:
                    sample = output_audio[:,SAMPLING_RATE*4:SAMPLING_RATE*(sample_length+4)]
                    torchaudio.save(os.path.join("tempDir","input.wav"), sample, SAMPLING_RATE)
                    st.pyplot(plot_specgram(sample,SAMPLING_RATE,title="Demixed {} {} secs Spectogram".format(i,sample_length)))
        loading.empty()

    st.write("--")

    do_comparsion = st.checkbox('Compare with another model')
    if do_comparsion:
        selection_compare = st.selectbox("Select model to compare",EXPERIMENTS,0)    
        # load model configurations
        model_path = "checkpoints/{}".format(selection_compare)
        audio_model, optimizer, loaded_data = load_model_and_optimizer(device,model_path)

        loading = st.empty()
        loading.write("loading....")

        output = {}

        audio_model.eval()

        # Disable gradient tracking
        with torch.no_grad():

            # Keep list predicted component chunks
            pred_chunk_list_dict = {i:list() for i in INSTRUMENTS}
        
            # Iterate through each chunk
            for segment_idx in range(len(input_waveform_chunks)):

                # Skip chunks without proper size
                input_data = input_waveform_chunks[segment_idx]
                if (input_data.shape[-1]<chunk_size): continue

                # Move chunk to GPU or alternative
                input_data = input_data.to(device)
                input_data = torch.reshape(input_data,(1,2,-1))

                for instr in INSTRUMENTS:

                    # Save pred if testing
                    pred = audio_model(input_data,instr)
                    pred_chunk_list_dict[instr].append(pred.cpu().detach())
            for i in INSTRUMENTS:
                output_audio = torch.cat(pred_chunk_list_dict[i],dim=-1).reshape((2,-1))
                torchaudio.save(os.path.join("tempDir","compare_{}.wav".format(i)), output_audio, SAMPLING_RATE,format="wav")
                st.write("Demixed {} output".format(i))
                st.audio(os.path.join("tempDir","compare_{}.wav".format(i)))
                show_waveform = st.checkbox('Show Comparison Demixed {} Waveform'.format(i))
                if show_waveform:
                    sample = output_audio[:,SAMPLING_RATE*4:SAMPLING_RATE*(sample_length+4)]
                    st.pyplot(plot_waveform(sample,SAMPLING_RATE,title="Demixed Comparison {} {} secs Waveform".format(i,sample_length)))
                show_spectogram = st.checkbox('Show Comparison Demixed {} Spectogram'.format(i))
                if show_spectogram:
                    if sample_length: 
                        st.pyplot(plot_specgram(sample,SAMPLING_RATE,title="Demixed Comparison {} {} secs Spectogram".format(i,sample_length)))
                    else:
                        sample = output_audio[:,SAMPLING_RATE*4:SAMPLING_RATE*(sample_length+4)]
                        st.pyplot(plot_specgram(sample,SAMPLING_RATE,title="Demixed Comparison {} {} secs Spectogram".format(i,sample_length)))
            loading.empty()