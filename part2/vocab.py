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
            # English Language Token
            self.word2id["<en>"] = 4
            # French Language Token
            self.word2id["<fr>"] = 5

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
        self,
        inputs: list[list[str]],
        targets: list[list[str]] | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Convert list of sentences (words) into tensor with necessary padding for
        shorter sentences.

        Args:
            inputs (list[list[str]]): list of sentences, where each sentence is represented as a
                list of words
            targets (list[list[str]], optional): list of target sentences, where each sentence is
                represented as a list of words
            device (torch.device): torch device

        Returns:
            sents_var (torch.Tensor): tensor of shape (batch_size, max_sentence_length) containing
                word indices
            tgts_var (torch.Tensor, optional): tensor of shape (batch_size, max_sentence_length)
                containing word indices for target sentences, only returned if targets is not None
        """
        if targets is None:
            src_sents = []
            for input_sent in inputs:
                # Concatenate input sentence with language tokens
                # Format: <src_lang> input_sentence <tgt_lang>
                combined_sent = ["<en>"] + input_sent + ["<fr>"]
                src_sents.append(combined_sent)

            word_ids = self.words2indices(src_sents)
            sents_t = pad_sents(word_ids, self["<pad>"])
            sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
            return sents_var

        src_sents = []
        tgt_sents = []
        for input_sent, target_sent in zip(inputs, targets):
            # Concatenate input and target sentences with language tokens
            # Format: <src_lang> input_sentence <tgt_lang> target_sentence
            combined_sent = ["<en>"] + input_sent + ["<fr>"] + target_sent
            src_sents.append(combined_sent)
            # The targets are the combined sentences shifted right by 1
            # We add </s> at the end to indicate the end of the target sentence
            # We also mask the src part by replacing it with <pad>
            tgt_combined_sent = combined_sent[1:] + ["</s>"]
            tgt_combined_sent[: len(input_sent) + 1] = ["<pad>"] * (len(input_sent) + 1)
            tgt_sents.append(tgt_combined_sent)

        word_ids = self.words2indices(src_sents)
        sents_t = pad_sents(word_ids, self["<pad>"])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)

        word_ids = self.words2indices(tgt_sents)
        tgts_t = pad_sents(word_ids, self["<pad>"])
        tgts_var = torch.tensor(tgts_t, dtype=torch.long, device=device)
        return sents_var, tgts_var

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


class Vocab:
    """Vocab for source and target languages."""

    def __init__(self, src_vocab: VocabEntry) -> None:
        """Initialize Vocab instance.

        Args:
            src_vocab (VocabEntry): vocabulary for both languages
        """
        self.src = src_vocab

    @staticmethod
    def build(sents: list[str]) -> "Vocab":
        """Build Vocabulary.

        Args:
            sents (list[str]): Source subwords provided by SentencePiece
        Returns:
            Vocab: Vocab object containing source and target VocabEntry
        """
        # assert len(sents) == len(tgt_sents)

        print("initialize vocabulary ..")
        # src = VocabEntry.from_corpus(sents, vocab_size, freq_cutoff)
        src = VocabEntry.from_subword_list(sents)

        return Vocab(src)

    def save(self, file_path: str) -> None:
        """Save Vocab to file as JSON.

        Args:
            file_path (str): file path to save vocab to
        """
        with open(file_path, "w") as f:
            json.dump(dict(src_word2id=self.src.word2id), f, indent=2)

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

        return Vocab(VocabEntry(src_word2id))

    def __repr__(self):
        """Representation of Vocab to be used when printing the object."""

        return f"Vocab {len(self.src)} words"


def get_vocab_list(
    file_paths: list[str], vocab_directory: str, vocab_size: int
) -> list[str]:
    """Use SentencePiece to tokenize and acquire list of unique subwords.

    Args:
        file_paths (list[str]): list of file paths where each filepath is a corpus
        sentencepiece_model_path (str): file path to save SentencePiece model
        vocab_size (int): desired vocabulary size

    Returns:
        sp_list (list[str]): list of unique subwords
    """

    def sentence_iterator():
        for file_path in file_paths:
            for line in open(file_path, "r"):
                yield line.strip()

    model_name = Path(vocab_directory, "spm.model")
    model_prefix = str(Path(vocab_directory, "spm"))
    # train the spm model
    spm.SentencePieceTrainer.Train(
        sentence_iterator=sentence_iterator(),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
    )
    # create an instance; this saves .model and .vocab files
    sp = spm.SentencePieceProcessor()
    # loads tgt.model or src.model
    sp.Load(str(model_name))
    # this is the list of subwords
    sp_list = [sp.IdToPiece(piece_id) for piece_id in range(sp.GetPieceSize())]
    return sp_list


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-source-file",
        type=Path,
        default=Path("..", "multi30k_data", "train.en"),
        help="Path to the train source file, containing source sentences.",
    )

    parser.add_argument(
        "--train-target-file",
        type=Path,
        default=Path("..", "multi30k_data", "train.fr"),
        help="Path to the train target file, containing gold-standard target sentences.",
    )

    parser.add_argument(
        "--vocab-directory",
        type=Path,
        default=Path("vocab"),
        help="Path to the output vocabulary directory.",
    )

    parser.add_argument(
        "--source-vocab-size",
        type=int,
        default=14563,
        help="Size of the source vocabulary.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    args.vocab_directory.mkdir(parents=True, exist_ok=True)

    sentences = get_vocab_list(
        [args.train_source_file, args.train_target_file],
        vocab_directory=str(Path(args.vocab_directory)),
        vocab_size=args.source_vocab_size,
    )

    vocab = Vocab.build(sentences)
    print(f"generated vocabulary, source {len(sentences)} words")

    print(f"vocab size: {len(vocab.src.word2id)}")

    vocab_file = str(Path(args.vocab_directory, "vocab.json"))
    vocab.save(vocab_file)
    print(f"vocabulary saved to {vocab_file}")
