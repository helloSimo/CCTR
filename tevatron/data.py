import random
from dataclasses import dataclass
from typing import List, Tuple

import datasets
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding

from .arguments import DataArguments
from .trainer import TevatronTrainer

import logging

logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            tokenizer: PreTrainedTokenizer,
            cache_dir: str,
            trainer: TevatronTrainer = None,
    ):
        data_files = data_args.train_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        self.train_data = load_dataset(data_args.dataset_name,
                                       data_args.dataset_language,
                                       data_files=data_files,
                                       cache_dir=cache_dir)[data_args.dataset_split]
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        passages = []
        group_positives = group['positive_passages']
        group_negatives = group['negative_passages']

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        if 'title' in pos_psg:
            pos_psg = pos_psg['title'] + self.data_args.passage_field_separator + pos_psg['text']
        else:
            pos_psg = pos_psg['text']
        passages.append(pos_psg)

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            if 'title' in neg_psg:
                neg_psg = neg_psg['title'] + self.data_args.passage_field_separator + neg_psg['text']
            else:
                neg_psg = neg_psg['text']
            passages.append(neg_psg)

        return group['query'], passages


class QueryDataset(Dataset):
    def __init__(self, data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        dataset = load_dataset(data_args.dataset_name,
                               data_args.dataset_language,
                               data_files=data_files,
                               cache_dir=cache_dir)[data_args.dataset_split]
        self.dataset = dataset.shard(data_args.encode_num_shard,
                                     data_args.encode_shard_index)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        example = self.dataset[item]
        return example['query_id'], example['query']


class CorpusDataset(Dataset):
    def __init__(self, data_args: DataArguments, cache_dir: str):
        data_files = data_args.encode_in_path
        if data_files:
            data_files = {data_args.dataset_split: data_files}
        dataset = load_dataset(data_args.dataset_name,
                               data_args.dataset_language,
                               data_files=data_files,
                               cache_dir=cache_dir)[data_args.dataset_split]
        self.dataset = dataset.shard(data_args.encode_num_shard,
                                     data_args.encode_shard_index)
        self.separator = data_args.passage_field_separator

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        example = self.dataset[item]
        if 'title' in example:
            text = example['title'] + self.separator + example['text']
        else:
            text = example['text']
        return example['docid'], text


@dataclass
class QPCollator:
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    tokenizer: PreTrainedTokenizer
    max_q_len: int = 32
    max_p_len: int = 128

    def __call__(self, features):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.tokenizer(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            truncation='only_first',
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            truncation='only_first',
            return_tensors="pt",
        )

        return q_collated, d_collated


@dataclass
class EncodeCollator:
    max_len: int
    tokenizer: PreTrainedTokenizer

    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        collated_features = self.tokenizer(
            text_features,
            padding='max_length',
            truncation='only_first',
            max_length=self.max_len,
            return_tensors="pt",
        )
        return text_ids, collated_features
