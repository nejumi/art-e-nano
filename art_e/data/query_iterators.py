import random
from typing import List, Optional

from datasets import Dataset, load_dataset

from art_e.data.types_enron import SyntheticQuery

HF_REPO_ID = "corbt/enron_emails_sample_questions"

# A few spot-checked synthetic queries that were ambiguous or contradicted by
# other source emails in the upstream ART-E example.
BAD_QUERIES = [49, 101, 129, 171, 208, 266, 327]


def load_synthetic_queries(
    split: str = "train",
    limit: Optional[int] = None,
    max_messages: Optional[int] = 1,
    shuffle: bool = False,
    seed: Optional[int] = None,
    exclude_known_bad_queries: bool = True,
) -> List[SyntheticQuery]:
    dataset: Dataset = load_dataset(HF_REPO_ID, split=split)  # type: ignore

    if max_messages is not None:
        dataset = dataset.filter(lambda x: len(x["message_ids"]) <= max_messages)

    if exclude_known_bad_queries:
        dataset = dataset.filter(lambda x: x["id"] not in BAD_QUERIES)

    if shuffle:
        dataset = dataset.shuffle(seed=seed)

    queries = [SyntheticQuery(**row) for row in dataset]  # type: ignore

    if max_messages is not None:
        queries = [query for query in queries if len(query.message_ids) <= max_messages]

    if shuffle:
        random.Random(seed).shuffle(queries)

    if limit is not None:
        return queries[:limit]
    return queries
