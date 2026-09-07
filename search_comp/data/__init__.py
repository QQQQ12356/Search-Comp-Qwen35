"""数据管线：语料构建、BM25 检索、SFT 数据、交互式轨迹、Dataset/collator。"""

from .retrieval import (
    BM25Retriever,
    build_corpus_from_hotpotqa,
    format_docs_as_reference,
)
from .sft_dataset import BeaconDataCollator, BeaconSFTDataset
from .interactive_dataset import InteractiveCollator, InteractiveSFTDataset
from .searchr1_dataset import SearchR1Collator, SearchR1SFTDataset
from .trajectory import (
    SEARCH_INSTRUCTION,
    build_sequence_ids,
    extract_search_query,
    format_information_block,
)

__all__ = [
    "BM25Retriever",
    "build_corpus_from_hotpotqa",
    "format_docs_as_reference",
    "BeaconSFTDataset",
    "BeaconDataCollator",
    "InteractiveSFTDataset",
    "InteractiveCollator",
    "SearchR1SFTDataset",
    "SearchR1Collator",
    "SEARCH_INSTRUCTION",
    "build_sequence_ids",
    "extract_search_query",
    "format_information_block",
]
