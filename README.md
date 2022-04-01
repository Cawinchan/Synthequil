# Synthequil
Audio synthesis project for Theory of Deep Learning SUTD

- Dataset: [Clotho Dataset](https://zenodo.org/record/4783391#.Yjyq-OdByUn) 
- FastText pre-trained model: [crawl-300d-2M-subword.zip](https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip): 2 million word vectors trained with subword information on Common Crawl (600B tokens).

## File Architecture 

```
├── raw_data
│     ├── Evaluation                       <- Evaluation music .wav
|     ├── Development                      <- Training music .wav
|     ├── clotho_captions_evaluation.csv   <- Evaluation caption labels
|     └── clotho_metadata_evaluation.csv   <- Evaluation source
| 
├── preproceseed_data                  
| 
├── pre-trained_fasttext                   <- Pre-trained fasttext model
| 
└──audio_synthesis.ipynb                   <- Notebook for audio generation
```

## How to install 

```python3
  pip install fasttext gensim pandas numpy
```

1. Place downloaded Clotho dataset in raw_data folder 
2. Place downloaded FastText model in pre-trained_fasttext
