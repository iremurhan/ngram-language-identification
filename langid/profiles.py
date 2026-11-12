"""Character n-gram rank profiles and the out-of-place distance.

Implements the N-gram-based text categorisation of Cavnar & Trenkle (1994):
a text is represented by its most frequent character n-grams in rank order,
and two profiles are compared by how far each n-gram is "out of place".
Pure standard library.
"""
import json
import unicodedata
from collections import Counter

PAD = "_"
APOSTROPHES = "'’"


def tokenize(text):
    """Lower-cased tokens made only of letters, combining marks and apostrophes.

    Combining marks (Unicode category M) are kept inside tokens: regex \\w
    does not match Devanagari or Arabic-script vowel signs, which would split
    Hindi and Urdu words in the middle.
    """
    tokens, current = [], []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat[0] in "LM" or (ch in APOSTROPHES and current):
            current.append(ch)
        elif current:
            tokens.append("".join(current).strip(APOSTROPHES).lower())
            current = []
    if current:
        tokens.append("".join(current).strip(APOSTROPHES).lower())
    return [t for t in tokens if t]


def ngram_counts(tokens, ns=(1, 2, 3)):
    """Count padded character n-grams, e.g. 'text' -> '_t', 'te', ..., 't_' for n = 2."""
    counts = Counter()
    for token, freq in Counter(tokens).items():
        for n in ns:
            padded = PAD + token + PAD * (n - 1)
            for i in range(len(padded) - n + 1):
                gram = padded[i:i + n]
                if gram.strip(PAD):
                    counts[gram] += freq
    return counts


def build_profile(text, ns=(1, 2, 3), top_n=300):
    """Top-n n-grams of a text in descending frequency (ties broken alphabetically)."""
    counts = ngram_counts(tokenize(text), ns)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [gram for gram, _ in ranked[:top_n]]


def out_of_place(document_profile, language_profile):
    """Sum of rank differences; an n-gram missing from the language profile costs len(language_profile)."""
    ranks = {gram: r for r, gram in enumerate(language_profile)}
    max_penalty = len(language_profile)
    return sum(abs(r - ranks[g]) if g in ranks else max_penalty
               for r, g in enumerate(document_profile))


class LanguageIdentifier:
    def __init__(self, profiles, ns=(1, 2, 3), top_n=300):
        self.profiles, self.ns, self.top_n = profiles, tuple(ns), top_n

    @classmethod
    def train(cls, texts_by_language, ns=(1, 2, 3), top_n=300):
        profiles = {lang: build_profile(text, ns, top_n) for lang, text in texts_by_language.items()}
        return cls(profiles, ns, top_n)

    def distances(self, text):
        doc = build_profile(text, self.ns, self.top_n)
        return {lang: out_of_place(doc, prof) for lang, prof in self.profiles.items()}

    def identify(self, text):
        d = self.distances(text)
        return min(d, key=d.get)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ns": list(self.ns), "top_n": self.top_n, "profiles": self.profiles}, f,
                      ensure_ascii=False, indent=1)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["profiles"], data["ns"], data["top_n"])
