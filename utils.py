import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
import nltk
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)

_NLTK_READY = False


def _ensure_nltk_data():
    global _NLTK_READY
    if _NLTK_READY:
        return
    for p in ("punkt", "averaged_perceptron_tagger", "wordnet", "omw-1.4"):
        try:
            if p == "punkt":
                nltk.data.find("tokenizers/punkt")
            elif p == "wordnet":
                nltk.data.find("corpora/wordnet")
            elif p == "averaged_perceptron_tagger":
                nltk.data.find("taggers/averaged_perceptron_tagger")
            elif p == "omw-1.4":
                nltk.data.find("corpora/omw-1.4")
        except LookupError:
            nltk.download(p, quiet=True)
    _NLTK_READY = True


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def _treebank_to_wordnet_pos(tag: str):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("R"):
        return wordnet.ADV
    if tag.startswith("N"):
        return wordnet.NOUN
    return None


_SKIP_SYNONYM = {
    "not",
    "no",
    "nor",
    "never",
    "neither",
    "nothing",
    "nobody",
    "nowhere",
    "n't",
}


def _synonym_replace(word: str, pos, rng: random.Random) -> str:
    if pos is None or not word.isalpha():
        return word
    if word.lower() in _SKIP_SYNONYM:
        return word
    syns = wordnet.synsets(word.lower(), pos=pos)
    if not syns:
        return word
    lemmas = []
    for s in syns[:3]:
        for lem in s.lemmas():
            w = lem.name().replace("_", " ")
            if w.lower() != word.lower():
                lemmas.append(w)
    if not lemmas:
        return word
    return rng.choice(lemmas)


def _inject_adjacent_typo(word: str, rng: random.Random) -> str:
    if len(word) <= 3 or not word.isalpha():
        return word
    if rng.random() >= 0.24:
        return word
    chars = list(word)
    i = rng.randint(0, len(chars) - 2)
    chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


# (lowercase_token, lowercase_next) -> expanded pair (lowercase)
_CONTRACTION_EXPAND = {
    ("do", "n't"): ("do", "not"),
    ("does", "n't"): ("does", "not"),
    ("did", "n't"): ("did", "not"),
    ("can", "n't"): ("can", "not"),
    ("could", "n't"): ("could", "not"),
    ("would", "n't"): ("would", "not"),
    ("should", "n't"): ("should", "not"),
    ("is", "n't"): ("is", "not"),
    ("are", "n't"): ("are", "not"),
    ("was", "n't"): ("was", "not"),
    ("were", "n't"): ("were", "not"),
    ("have", "n't"): ("have", "not"),
    ("has", "n't"): ("has", "not"),
    ("had", "n't"): ("had", "not"),
    ("won", "n't"): ("will", "not"),
}

_EXPAND_TO_CONTR = {v: k for k, v in _CONTRACTION_EXPAND.items()}


def _cap_first(s: str, like: str) -> str:
    if like and like[0].isupper():
        return s[:1].upper() + s[1:] if s else s
    return s


def _apply_contraction_perturbation(tokens, rng: random.Random):
    """Expand or contract common contraction / expansion sites (stronger perturbation)."""
    out = []
    i = 0
    n = len(tokens)
    while i < n:
        if i + 1 < n:
            pair = (tokens[i].lower(), tokens[i + 1].lower())
            if pair in _CONTRACTION_EXPAND and rng.random() < 0.48:
                exp_a, exp_b = _CONTRACTION_EXPAND[pair]
                if rng.random() < 0.62:
                    a = _cap_first(exp_a, tokens[i])
                    out.extend([a, exp_b])
                else:
                    out.extend([tokens[i], tokens[i + 1]])
                i += 2
                continue
            epair = (tokens[i].lower(), tokens[i + 1].lower())
            if epair in _EXPAND_TO_CONTR and rng.random() < 0.48:
                c0, c1 = _EXPAND_TO_CONTR[epair]
                if rng.random() < 0.62:
                    c0 = _cap_first(c0, tokens[i])
                    out.extend([c0, c1])
                else:
                    out.extend([tokens[i], tokens[i + 1]])
                i += 2
                continue
        out.append(tokens[i])
        i += 1
    return out


def custom_transform(example, idx=0):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    _ensure_nltk_data()
    rng = random.Random(int(idx) + 1000003 * int(example["label"]))

    text = example["text"]
    tokens = word_tokenize(text)
    if not tokens:
        return example

    tokens = _apply_contraction_perturbation(tokens, rng)

    tagged = pos_tag(tokens)
    new_tokens = []
    for word, tag in tagged:
        wn_pos = _treebank_to_wordnet_pos(tag)
        if wn_pos is None:
            new_tokens.append(word)
            continue
        if wn_pos == wordnet.NOUN:
            p_syn = 0.14 if len(word) > 2 else 0.0
        else:
            p_syn = 0.28
        if rng.random() < p_syn:
            repl = _synonym_replace(word, wn_pos, rng)
            new_tokens.append(repl)
        else:
            new_tokens.append(word)

    for i in range(len(new_tokens) - 1):
        if rng.random() < 0.11:
            new_tokens[i], new_tokens[i + 1] = new_tokens[i + 1], new_tokens[i]

    new_tokens = [_inject_adjacent_typo(w, rng) for w in new_tokens]

    detok = TreebankWordDetokenizer()
    out_text = detok.detokenize(new_tokens)
    if out_text and rng.random() < 0.14:
        hedges = ("I guess ", "In a way, ", "Sort of ", "Kind of ")
        if len(out_text) > 1 and out_text[0].isalpha():
            out_text = rng.choice(hedges) + out_text[0].lower() + out_text[1:]
        else:
            out_text = rng.choice(hedges) + out_text
    example["text"] = out_text

    ##### YOUR CODE ENDS HERE ######

    return example
