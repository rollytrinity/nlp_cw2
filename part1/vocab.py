import argparse
import json
from collections import Counter
from itertools import chain
from pathlib import Path

import sentencepiece as spm
import torch

from utils import pad_sents


class VocabEntry:
    """Vocabulary Entry, containing either src or tgt language terms."""

    def __init__(self, word2id: dict[str, int] | None = None):
        """Initialize VocabEntry Instance.

        Args:
            word2id (dict[str, int], optional): dictionary mapping words to their ids.
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = {}
            # Pad Token
            self.word2id["<pad>"] = 0
            # Start of Sentence Token
            self.word2id["<s>"] = 1
            # End of Sentence Token
            self.word2id["</s>"] = 2
            # Unknown Word Token
            self.word2id["<unk>"] = 3

        self.unk_id = self.word2id["<unk>"]
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word: str) -> int:
        """Retrieve word's index.

        Note if the word is out of the vocabulary, returns the index for the unk token

        Args:
            word (str): word to look up.

        Returns:
            index (int): index of word
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word: str) -> bool:
        """Check if word is captured by VocabEntry.

        Args:
            word (str): word to look up

        Returns:
            contains (bool): whether word is contained
        """
        return word in self.word2id

    def __setitem__(self, key: str, value: int):
        """Raise error, if one tries to edit the VocabEntry."""
        raise ValueError("vocabulary is readonly")

    def __len__(self) -> int:
        """Compute number of words in VocabEntry.

        Returns:
            length (int): number of words in VocabEntry
        """
        return len(self.word2id)

    def __repr__(self) -> str:
        """Representation of VocabEntry to be used when printing the object."""
        return f"Vocabulary[size={len(self)}]"

    def id2word(self, word_id: int) -> str:
        """Return mapping of index to word.

        Args:
            word_id (int): word index

        Returns:
            word (str): word corresponding to index
        """
        return self.id2word[word_id]

    def add(self, word: str) -> int:
        """Add word to VocabEntry, if it is previously unseen.

        Args:
            word (str): word to add to VocabEntry

        Returns:
            index (int): index that the word has been assigned
        """
        if word not in self:
            word_id = self.word2id[word] = len(self)
            self.id2word[word_id] = word
            return word_id
        else:
            return self[word]

    def words2indices(self, sents: list[list[str]]) -> list[list[int]] | list[int]:
        """Convert list of words or list of sentences of words into list or list of list of indices.

        Args:
            sents (list[str] or list[list[str]]): sentence(s) in words

        Returns:
            word_ids (list[int] or list[list[int]]): sentence(s) in indices
        """
        if isinstance(sents[0], list):
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids: list[int]) -> list[str]:
        """Convert list of indices into words.

        Args:
            word_ids (list[int]): list of word indices

        Returns:
            words (list[str]): list of words
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(
        self, sents: list[list[str]], device: torch.device
    ) -> torch.Tensor:
        """Convert list of sentences (words) into tensor with necessary padding for
        shorter sentences.

        Args:
            sents (list[list[str]]): list of sentences, where each sentence is represented as
                a list of words
            device (torch.device): torch device

        Returns:
            sents_var (torch.Tensor): tensor of shape (max_sentence_length, batch_size)
        """
        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self["<pad>"])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)

    @staticmethod
    def from_corpus(corpus: list[str], size: int, freq_cutoff: int = 2) -> "VocabEntry":
        """Given a corpus construct a Vocab Entry.

        Args:
            corpus (list[str]): list of sentences, where each sentence is represented as a list of words
            size (int): size of vocabulary
            freq_cutoff (int, optional): frequency cutoff for words to be included in vocabulary.

        Returns:
            VocabEntry: VocabEntry created from corpus
        """
        vocab_entry = VocabEntry()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        print(
            "number of word types: {}, number of word types w/ frequency >= {}: {}".format(
                len(word_freq), freq_cutoff, len(valid_words)
            )
        )
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[
            :size
        ]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry

    @staticmethod
    def from_subword_list(subword_list: list[str]) -> "VocabEntry":
        """Given a list of subwords construct a Vocab Entry.
        Args:
            subword_list (list[str]): list of subwords

        Returns:
            VocabEntry: VocabEntry created from subword list
        """
        vocab_entry = VocabEntry()
        for subword in subword_list:
            vocab_entry.add(subword)
        return vocab_entry


class Vocab(object):
    """Vocab for source and target languages."""

    def __init__(self, src_vocab: VocabEntry, tgt_vocab: VocabEntry) -> None:
        """Initialize Vocab instance.

        Args:
            src_vocab (VocabEntry): vocabulary for source language
            tgt_vocab (VocabEntry): vocabulary for target language
        """
        self.src = src_vocab
        self.tgt = tgt_vocab

    @staticmethod
    def build(src_sents: list[str], tgt_sents: list[str]) -> "Vocab":
        """Build Vocabulary.

        Args:
            src_sents (list[str]): Source subwords provided by SentencePiece
            tgt_sents (list[str]): Target subwords provided by SentencePiece
        Returns:
            Vocab: Vocab object containing source and target VocabEntry
        """
        # assert len(src_sents) == len(tgt_sents)

        print("initialize source vocabulary ..")
        # src = VocabEntry.from_corpus(src_sents, vocab_size, freq_cutoff)
        src = VocabEntry.from_subword_list(src_sents)

        print("initialize target vocabulary ..")
        # tgt = VocabEntry.from_corpus(tgt_sents, vocab_size, freq_cutoff)
        tgt = VocabEntry.from_subword_list(tgt_sents)

        return Vocab(src, tgt)

    def save(self, file_path: str) -> None:
        """Save Vocab to file as JSON.

        Args:
            file_path (str): file path to save vocab to
        """
        with open(file_path, "w") as f:
            json.dump(
                dict(src_word2id=self.src.word2id, tgt_word2id=self.tgt.word2id),
                f,
                indent=2,
            )

    @staticmethod
    def load(file_path: str) -> "Vocab":
        """Load vocabulary from JSON.

        Args:
            file_path (str): file path to load vocab from

        Returns:
            Vocab: Vocab object containing source and target VocabEntry
        """
        entry = json.load(open(file_path, "r"))
        src_word2id = entry["src_word2id"]
        tgt_word2id = entry["tgt_word2id"]

        return Vocab(VocabEntry(src_word2id), VocabEntry(tgt_word2id))

    def __repr__(self):
        """Representation of Vocab to be used when printing the object."""

        return f"Vocab(source {len(self.src)} words, target {len(self.tgt)} words)"


def get_vocab_list(file_path: str, source: str, vocab_size: int) -> list[str]:
    """Use SentencePiece to tokenize and acquire list of unique subwords.

    Args:
        file_path (str): file path to corpus
        source (str): tgt or src
        vocab_size (int): desired vocabulary size

    Returns:
        sp_list (list[str]): list of unique subwords
    """
    # train the spm model
    spm.SentencePieceTrainer.Train(
        input=file_path, model_prefix=source, vocab_size=vocab_size
    )
    # create an instance; this saves .model and .vocab files
    sp = spm.SentencePieceProcessor()
    # loads tgt.model or src.model
    sp.Load("{}.model".format(source))
    # this is the list of subwords
    sp_list = [sp.IdToPiece(piece_id) for piece_id in range(sp.GetPieceSize())]
    return sp_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-source-file",
        type=Path,
        default=Path("multi30k_data", "train.en"),
        help="Path to the train source file, containing source sentences.",
    )

    parser.add_argument(
        "--train-target-file",
        type=Path,
        default=Path("multi30k_data", "train.fr"),
        help="Path to the train target file, containing gold-standard target sentences.",
    )

    parser.add_argument(
        "--vocab-file",
        type=Path,
        default=Path("vocab", "vocab.json"),
        help="Path to the output vocabulary file.",
    )

    parser.add_argument(
        "--source-vocab-size",
        type=int,
        default=7000,
        help="Size of the source vocabulary.",
    )

    parser.add_argument(
        "--target-vocab-size",
        type=int,
        default=8000,
        help="Size of the target vocabulary.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"read in source sentences: {args.train_source_file}")
    print(f"read in target sentences: {args.train_target_file}")

    src_sents = get_vocab_list(
        args.train_source_file, source="src", vocab_size=args.source_vocab_size
    )

    tgt_sents = get_vocab_list(
        args.train_target_file, source="tgt", vocab_size=args.target_vocab_size
    )

    vocab = Vocab.build(src_sents, tgt_sents)
    print(
        f"generated vocabulary, source {len(src_sents)} words, target {len(tgt_sents)} words"
    )

    if not args.vocab_file.parent.exists():
        args.vocab_file.parent.mkdir(parents=True, exist_ok=True)

    vocab.save(args.vocab_file)
    print(f"vocabulary saved to {args.vocab_file}")
