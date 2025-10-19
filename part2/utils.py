import math
from typing import Generator

import nltk
import numpy as np
import sentencepiece as spm

nltk.download("punkt_tab")


def pad_sents(sents: list[list[str]], pad_token: str) -> list[list[str]]:
    """Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.

    Args:
        sents (list[list[str]]): list of sentences, where each sentence is represented as
            a list of words
        pad_token (str): padding token

    Returns:
        sents_padded (list[list[str]]): list of sentences where sentences shorter
            than the max length sentence are padded out with the pad_token, such that
            each sentences in the batch now has equal length.
    """

    max_len = max(len(sent) for sent in sents)
    sents_padded = []
    for sent in sents:
        num_pads = max_len - len(sent)
        sents_padded.append(sent + [pad_token] * num_pads)

    return sents_padded


def read_corpus(
    src_file_path: str, tgt_file_path: str, model_path: str, vocab_size: int = 2500
) -> list[list[str]]:
    """Read file, where each sentence is dilineated by a `\n`.

    Args:
        src_file_path (str): path to file containing src corpus
        tgt_file_path (str): path to file containing tgt corpus
        model_path (str): path to SentencePiece model
        vocab_size (int): number of unique subwords in vocabulary when reading and tokenizing

    Returns:
        data (list[list[str]]): list of sentences, where each sentence is represented as a list of words
    """
    data = []
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    with (
        open(src_file_path, "r", encoding="utf8") as src_file,
        open(tgt_file_path, "r", encoding="utf8") as tgt_file,
    ):
        for src_line, tgt_line in zip(src_file, tgt_file):
            src_subword_tokens = sp.encode_as_pieces(src_line)
            tgt_subword_tokens = sp.encode_as_pieces(tgt_line)

            data.append((src_subword_tokens, tgt_subword_tokens))

    return data


def batch_iter(
    data: list[tuple[list[str], list[str]]], batch_size: int, shuffle: bool = False
) -> Generator:
    """Yield batches of source and target sentences reverse sorted by length (largest to smallest).

    Args:
        data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
        batch_size (int): batch size
        shuffle (boolean): whether to randomly shuffle the dataset

    Yields:
            src_sents (list[list[str]]): list of source sentences in the
            tgt_sents (list[list[str]]): list of target sentences in the
                batch, sorted in the order of decreasing source sentence length.
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size : (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
