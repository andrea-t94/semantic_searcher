''' Train a specific version of the model, identified by MODEL_PARAMS'''

from pathlib import Path
import sys
import os
import logging
import json

from datasets import load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

#FIXME
sys.path.append(str(Path(__file__).parents[1]))
from constants import DATASETS_PATH, MODELS_PATH, MODEL_PARAMS

# Environs
SAMPLING = os.getenv('SAMPLING', True)
OVERRIDE = os.getenv('OVERRIDE', False)
VERSION = os.getenv('MODEL_VERSION')
CURRENT_MODELS_PATH = f"{MODELS_PATH}/v{VERSION}"

def main():
    # Load the datasets for training
    logging.info(f"Data loading for training and evaluation")
    train_dataset = load_from_disk(f"{DATASETS_PATH}/v{MODEL_PARAMS['dataset_version']}/train")
    if SAMPLING:
        logging.info(f"Data down sampled")
        train_dataset = train_dataset.select(range(1000)) #necessary to work on local CPU
    eval_dataset = load_from_disk(f"{DATASETS_PATH}/v{MODEL_PARAMS['dataset_version']}/eval")

    # Load a model to finetune, chosen the smallest one among the ones trained on MS-MARCO
    logging.info(f"Model inizialization")
    model = SentenceTransformer(MODEL_PARAMS['baseline'])
    # https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py#L38C9-L38C23
    model.max_seq_length=MODEL_PARAMS['max_seq_lenght']
    model.tokenizer.model_max_length=MODEL_PARAMS['max_seq_lenght']

    # Define a loss function
    loss = MultipleNegativesRankingLoss(model)

    # Specify training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=f"{CURRENT_MODELS_PATH}/checkpoints",
        num_train_epochs=MODEL_PARAMS['epochs'],
        per_device_train_batch_size=MODEL_PARAMS['batch_size'],
        per_device_eval_batch_size=MODEL_PARAMS['batch_size'],
        gradient_accumulation_steps=4,
        fp16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # Losses using "in-batch negatives" benefit from no duplicates
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=200,
    )

    # Create an evaluator & evaluate the base model
    logging.info(f"Evaluator inizialization")
    dev_evaluator = TripletEvaluator(
        anchors=eval_dataset["query"],
        positives=eval_dataset["positive_passages"],
        negatives=eval_dataset["negative_passages"],
        name="trec-eval",
    )

    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset= train_dataset, #ds, #train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        evaluator=dev_evaluator,
    )
    logging.info(f"Training start")
    trainer.train()

    # save model and its config
    model.save_pretrained(CURRENT_MODELS_PATH)
    with open(f"{CURRENT_MODELS_PATH}/card.json", "w") as outfile:
        json.dump(MODEL_PARAMS, outfile)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    if not VERSION:
        logging.info(f"Missing required input arg VERSION")
    # check if model version already exists and, in case OVERRIDE is set to False, skip the processing
    if os.path.exists(CURRENT_MODELS_PATH) and not OVERRIDE:
        logging.info(f"Model v{VERSION} already exists in: {CURRENT_MODELS_PATH}")
    else:
        main()

