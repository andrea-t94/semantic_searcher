from pathlib import Path

DATASETS_PATH = f'{Path(__file__).parents[1]}/artifacts/datasets'
MODELS_PATH = f'{Path(__file__).parents[1]}/artifacts/models'
BASELINE_MODEL_PATH = f'{Path(__file__).parents[0]}/serve/model'

MODEL_PARAMS = {
    'baseline': 'sentence-transformers/msmarco-distilbert-cos-v5',
    'epochs': 1,
    'batch_size': 8,
    'max_seq_lenght': 64, #optimized accordingly to EDA
    'dataset_version': 1 #it has to exist
}