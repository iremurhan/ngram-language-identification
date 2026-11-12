"""Snippet-level accuracy: how many words does the method need?"""
from collections import Counter, defaultdict


def snippets(text, n_words):
    """Consecutive, non-overlapping chunks of n_words whitespace-separated words."""
    words = text.split()
    return [" ".join(words[i:i + n_words]) for i in range(0, len(words) - n_words + 1, n_words)]


def evaluate(identifier, test_texts, lengths=(1, 3, 5, 10, 20, 50)):
    """Return {length: {"accuracy": ..., "per_language": {...}, "confusions": Counter}}."""
    results = {}
    for n in lengths:
        correct, total = defaultdict(int), defaultdict(int)
        confusions = Counter()
        for lang, text in test_texts.items():
            for snip in snippets(text, n):
                pred = identifier.identify(snip)
                total[lang] += 1
                correct[lang] += pred == lang
                if pred != lang:
                    confusions[(lang, pred)] += 1
        results[n] = {
            "accuracy": sum(correct.values()) / max(sum(total.values()), 1),
            "per_language": {l: correct[l] / total[l] for l in total},
            "confusions": confusions,
        }
    return results
