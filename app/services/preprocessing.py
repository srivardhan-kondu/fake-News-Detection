import re
import string
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


STOP_WORDS = set(ENGLISH_STOP_WORDS)


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"https?://\S+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def remove_duplicate_tokens(tokens: list[str]) -> list[str]:
    seen = set()
    unique_tokens = []
    for token in tokens:
        if token not in seen:
            unique_tokens.append(token)
            seen.add(token)
    return unique_tokens


def preprocess_text(text: str) -> str:
    normalized = normalize_text(text)
    tokens = tokenize(normalized)
    filtered_tokens = [token for token in tokens if token not in STOP_WORDS and token not in string.punctuation]
    filtered_tokens = remove_duplicate_tokens(filtered_tokens)
    return " ".join(filtered_tokens)


def get_word_frequency(text: str, top_n: int = 10) -> dict[str, int]:
    tokens = [token for token in tokenize(normalize_text(text)) if token not in STOP_WORDS]
    counts = Counter(tokens)
    return dict(counts.most_common(top_n))
