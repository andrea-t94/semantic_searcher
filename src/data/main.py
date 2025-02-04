''' Prepare dataset for training and testing against current best model '''

import os
import sys
import logging
from pathlib import Path

from datasets import load_dataset

from utils import process_dataset
#FIXME
sys.path.append(str(Path(__file__).parents[1]))
from constants import DATASETS_PATH

VERSION = os.getenv('DATASET_VERSION')
OVERRIDE = bool(os.getenv('OVERRIDE', False))
TEST_SPLIT = os.getenv('DATASET_TEST_SPLIT_RATIO', 0.05)
CURRENT_DATASETS_PATH = f"{DATASETS_PATH}/v{VERSION}"

def main():
    # get raw data
    logging.info(f"Getting raw data from trec-product-search/Product-Search-Triples")
    train_dataset_raw = load_dataset("trec-product-search/Product-Search-Triples", split="train")
    dev_dataset_raw = load_dataset("trec-product-search/Product-Search-Triples", split="dev")
    split_dataset = dev_dataset_raw.train_test_split(test_size=TEST_SPLIT)

    # data processing
    logging.info(f"Processing raw data for training and evaluation")
    train_dataset = process_dataset(train_dataset_raw)
    # TODO: improper called "train" since it's the dev split. The test split contains no passages, thus the need to split it
    eval_dataset = process_dataset(split_dataset['train'])
    test_dataset = split_dataset['test'] #no need to process it, will be exploded on the fly during evaluation

    # save datasets with its version
    logging.info(f"Saving datasets in {CURRENT_DATASETS_PATH}")
    train_dataset.save_to_disk(f"{CURRENT_DATASETS_PATH}/train")
    eval_dataset.save_to_disk(f"{CURRENT_DATASETS_PATH}/eval")
    test_dataset.save_to_disk(f"{CURRENT_DATASETS_PATH}/test")

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    # check if dataset version already exists and, in case OVERRIDE is set to False, skip the processing
    if not VERSION:
        logging.info(f"Missing required input arg VERSION")
    if os.path.exists(CURRENT_DATASETS_PATH) and not OVERRIDE:
        logging.info(f"Dataset v{VERSION} already exists in: {CURRENT_DATASETS_PATH}")
    else:
        main()