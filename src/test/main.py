''' Test a specific version of the model, identified by MODEL_PARAMS. If better than baseline, substitute the baseline '''

from pathlib import Path
import sys
import os
import logging
import json

from datasets import load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    export_dynamic_quantized_onnx_model
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator

#FIXME
sys.path.append(str(Path(__file__).parents[1]))
from constants import DATASETS_PATH, MODELS_PATH, MODEL_PARAMS, BASELINE_MODEL_PATH
from utils import process_test_dataset

# Environs
SAMPLING = os.getenv('SAMPLING', True)
VERSION = os.getenv('MODEL_VERSION')
CURRENT_MODELS_PATH = f"{MODELS_PATH}/v{VERSION}"

def main():
    # Load the datasets for training
    logging.info(f"Data loading for training and evaluation")
    test_dataset = load_from_disk(f"{DATASETS_PATH}/v{MODEL_PARAMS['dataset_version']}/test")
    if SAMPLING:
        logging.info(f"Data down-sampled")
        test_dataset = test_dataset.select(range(100)) #necessary to work on local CPU

    # Data processing
    logging.info(f"Data processing to work with InformationRetrievealEvaluator")
    queries, relevant_docs, corpus = process_test_dataset(test_dataset)

    # Load the model trained (identified by MODEL_PARAMS) and the baseline
    model = SentenceTransformer(f'{CURRENT_MODELS_PATH}')
    try:
        baseline = SentenceTransformer(BASELINE_MODEL_PATH)
    except:
        logging.info(f"No local model found, downloading baseline from hub")
        baseline = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')

    # Define the evaluator
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="trec-test",
    )

    # Evaluate
    logging.info(f"Evaluation - model v{VERSION}")
    results_model = ir_evaluator(model)
    logging.info(f"Evaluation - baseline model")
    results_baseline = ir_evaluator(baseline)

    # In case new model is better than baseline in NDCG, set this as new baseline
    if results_model['trec-test_cosine_ndcg@10'] > results_baseline['trec-test_cosine_ndcg@10']:
        logging.info(f"Overwriting baseline model with model v{VERSION}")
        model.save_pretrained(BASELINE_MODEL_PATH)
        with open(f"{BASELINE_MODEL_PATH}/card.json", "w") as outfile:
            json.dump(results_model, outfile)
    else:
        logging.info(f"Baseline model better than model v{VERSION}")



if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    if not VERSION:
        logging.info(f"Missing required input arg VERSION")
    else:
        main()

