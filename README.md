# Synthequil

Challenge: Can we take an audio file with 4 instuments played together and output 4 audio files of each seperate instrument?

- Dataset: [MUSDB18-HQ](https://zenodo.org/record/3338373#.YknC3DURW3A) 
  - It consists of a total of 150 full-track songs of different styles and includes both the stereo mixtures and the original sources, divided between a training subset and a test subset.
- Reference: [Hybrid Spectrogram and Waveform Source Separation](https://arxiv.org/pdf/2111.03600.pdf)

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

Run the following command:

```python3
  pip install torch torchaudio pandas numpy torchvision glob matplotlib alive-progress IPython
```

## Usage instructions

The full list of parameters are as follows:
```
usage: main.py [-h] [--dataset-dir [dataset root dir]] [--log-dir [root log dir]] [--test [True/False]] [--custom-test-dir [custom test input folder path]]
               [--train-checkpoint-dir [root directory to store checkpoints]] [--test-output-dir [output root dir to store test outputs]] [--model [path to model checkpoint]]
               [--epoch-count [number of epochs to train model]] [--learning-rate [learning rate]] [--block-count [number of downsampling/upsampling blocks]] [--dropout [True/False]]
               [--dropout-proba [dropout probability]] [--scale-pow [power to scale inputs by]]

Training/testing program for Audio Demixing

optional arguments:
  -h, --help            show this help message and exit
  --dataset-dir [dataset root dir]
                        Root directory for dataset, containing train and test folders; ignored if custom input is specified for test mode
  --log-dir [root log dir]
                        Root directory to store training/testing logs (default: ./logs)
  --test [True/False]   Toggle test mode
  --custom-test-dir [custom test input folder path]
                        Custom input folder for testing
  --train-checkpoint-dir [root directory to store checkpoints]
                        Root directory to store checkpoints of model during training (default: ./checkpoints)
  --test-output-dir [output root dir to store test outputs]
                        Root directory to store test outputs; ignored if training (default: ./test_output)
  --model [path to model checkpoint]
                        File path to model checkpoint for testing or continuing training
  --epoch-count [number of epochs to train model]
                        Number of epochs by which to train model (default: 50)
  --learning-rate [learning rate]
                        Learning rate of model; ignored if loading model (default: 0.01)
  --block-count [number of downsampling/upsampling blocks]
                        Number of downsampling and of upsampling blocks in model; ignored if loading model (default: 1)
  --dropout [True/False]
                        Toggle dropout for training; ignored if loading model
  --dropout-proba [dropout probability]
                        Probability used for dropout layers; ignored if --dropout is not used as well (default: 0.2)
  --scale-pow [power to scale inputs by]
                        Exponent to scale input samples by, while preserving amplitude signs (default: 0.5)
```

Note that:
- For training, --dataset-dir must be used
- For testing, either --dataset-dir or --custom-test-dir, and --model must be used

Suppose that you would like to train the main configuration with:
- 50 epochs
- Dropout with probability 0.2
- Learning rate 0.001

Then you can run:
```python3
  python main.py --dataset-dir [your dataset root directory here]
```

Suppose you also want to test the same configuration for the epoch 50 checkpoint, then run:
```python3
  python main.py --model [your model path here] --dataset-dir [your dataset root directory here]
```
By default:
- Models are given an ID "b\[number of sampling blocks\]_lr\[learning rate\]_d\[dropout probability or N (if disabled)\]_scl\[power to scale inputs by\]"
- Logs are stored in ./logs/train/\[model id\]/\[time of training start\] for training, or ./logs/test/\[model id\]/\[time of test start\] for testing
- Test outputs are stored in ./test_output/\[song index\] where song index is arbitrarily decided
- Checkpoints are stored in ./checkpoints/\[model_id\]/

For custom test inputs, provide a folder with all songs to be tested, where the folder contains 1 folder for each song, and each song contains the mixture track (mixture.wav) and 4 component tracks (bass.wav, drums.wav, other.wav, vocals.wav):
```
├── Root folder    <- Provide directory for this
│     ├── Song folder
|           ├ mixture.wav
|           ├ bass.wav
|           ├ drums.wav
|           ├ other.wav
|           ├ vocals.wav
'''

Additionally, when testing, the model outputs each song in a separate folder, containing both predicted (with filename prepended with "_pred") and target component tracks.