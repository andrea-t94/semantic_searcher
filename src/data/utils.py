from typing import Dict, List, TypedDict, Optional
from datasets import Dataset
from dataclasses import dataclass


class Passage(TypedDict):
    """Represents a passage in the dataset with its title."""
    title: str


class DatasetExample(TypedDict):
    """Represents a raw example from the dataset before processing."""
    query: str
    query_id: str
    positive_passages: List[Passage]
    negative_passages: List[Passage]


class ProcessedExample(TypedDict):
    """Represents a processed example with validation flag."""
    query: str
    positive_passages: str
    negative_passages: str
    valid: bool


def process_dataset(dataset: Dataset) -> Dataset:
    """
    Process a dataset containing query-passage triplets, filtering out invalid examples
    and simplifying the passage structure.

    This function processes each example in the dataset by:
    1. Validating the presence of positive and negative passages
    2. Extracting the first positive and negative passage titles
    3. Filtering out invalid examples
    4. Removing unnecessary columns

    Args:
        dataset (Dataset): A HuggingFace dataset containing examples with queries
                         and associated positive and negative passages.

    Returns:
        Dataset: A processed dataset containing only valid examples with simplified structure.
                Each example contains:
                - query: The original query text
                - positive_passages: Title of the first positive passage
                - negative_passages: Title of the first negative passage

    Raises:
        ValueError: If the dataset structure is invalid or missing required fields
    """

    def process_triplet(example: DatasetExample) -> ProcessedExample:
        """
        Process a single example from the dataset, validating and extracting relevant information.

        Args:
            example (DatasetExample): Raw example from the dataset

        Returns:
            ProcessedExample: Processed example with validation flag
        """

        has_valid_data = (
                isinstance(example['positive_passages'], list) and
                isinstance(example['negative_passages'], list) and
                len(example['positive_passages']) > 0 and
                len(example['negative_passages']) > 0
        )

        if has_valid_data:
            processed = {
                'query': example['query'],
                'positive_passages': example['positive_passages'][0]['title'],
                'negative_passages': example['negative_passages'][0]['title'],
                'valid': True
            }
        else:
            processed = {
                'query': '',  # Empty placeholder
                'positive_passages': '',
                'negative_passages': '',
                'valid': False
            }

        return processed

    try:
        # Apply the processing function to the dataset
        processed_dataset = dataset.map(
            process_triplet,
            desc="Processing triplets"
        )

        # Filter to keep only valid examples
        final_dataset = processed_dataset.filter(
            lambda x: x['valid'],
            desc="Filtering valid examples"
        )

        # Remove unnecessary columns
        columns_to_remove = ['query_id', 'valid']
        final_dataset = final_dataset.remove_columns(columns_to_remove)

        # Verify the final dataset structure
        if len(final_dataset) == 0:
            raise ValueError("No valid examples found in the dataset after processing")

        return final_dataset

    except Exception as e:
        raise ValueError(f"Error processing dataset: {str(e)}")