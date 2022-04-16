# Synthequil
Audio Demixing project for Theory of Deep Learning SUTD

- Dataset: [MUSDB18-HQ](https://zenodo.org/record/3338373#.YknC3DURW3A) 
  - It consists of a total of 150 full-track songs of different styles and includes both the stereo mixtures and the original sources, divided between a training subset and a test subset.
-Reference: [Hybrid Spectrogram and Waveform Source Separation](https://arxiv.org/pdf/2111.03600.pdf)

## File Architecture 

```
├── raw_data
│     ├── test                       <- test music .wav
|     ├── train                      <- Training music .wav
| 
├── audio_splitter.ipynb             <- Notebook for audio spliting
| 
└── exploration_code.ipynb           <- Notebook for exploration 
```

## How to install 

```python3
  pip install torch torchaudio pandas numpy torchvision glob matplotlib librosa IPython
```

1. Place downloaded MUSDB18-HQ dataset in raw_data folder 
