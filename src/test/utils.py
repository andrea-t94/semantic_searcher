from collections import defaultdict
from typing import Dict, List

from typing import Dict, List, Tuple, TypedDict
from collections import defaultdict
from datasets import Dataset


def process_test_dataset(dataset: Dataset) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, str]]:
    """
    Process a HuggingFace dataset containing query-document pairs and their relevance information.

    This function extracts queries, their relevant documents, and builds a document corpus from
    a dataset typically used for information retrieval tasks.

    Args:
        dataset (Dataset): A HuggingFace dataset containing samples with queries and associated
                         positive and negative passages.

    Returns:
        Tuple containing:
            - Dict[str, str]: Mapping of query IDs to query texts
            - Dict[str, List[str]]: Mapping of query IDs to lists of relevant document IDs
            - Dict[str, str]: Mapping of document IDs to their titles (corpus)

    Raises:
        KeyError: If required fields are missing from the dataset
        TypeError: If the dataset structure doesn't match expected types
    """
    # Initialize our output dictionaries
    queries: Dict[str, str] = {}
    relevant_docs: Dict[str, List[str]] = defaultdict(list)
    corpus: Dict[str, str] = {}

    try:
        # Process each split in the dataset
        for sample in dataset:
            # Cast the sample to our TypedDict for better type checking
            typed_sample = sample

            query_id = typed_sample['query_id']
            query = typed_sample['query']
            positives = typed_sample['positive_passages']
            negatives = typed_sample['negative_passages']

            # Store query
            queries[query_id] = query

            # Process positive documents
            for pos_doc in positives:
                doc_id = pos_doc['docid']
                title = pos_doc['title']
                relevant_docs[query_id].append(doc_id)
                corpus[doc_id] = title

            # Process negative documents (for corpus only)
            for neg_doc in negatives:
                doc_id = neg_doc['docid']
                title = neg_doc['title']
                corpus[doc_id] = title

    except KeyError as e:
        raise KeyError(f"Missing required field in dataset: {str(e)}")
    except TypeError as e:
        raise TypeError(f"Dataset structure doesn't match expected types: {str(e)}")

    return queries, relevant_docs, corpus