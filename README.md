# N-gram Language Identification

A from-scratch implementation of N-gram-based text categorisation (Cavnar & Trenkle, 1994) for identifying the language of a text among nine languages: Dutch, English, German, Hindi, Portuguese, Russian, Serbian, Turkish and Urdu. Pure Python standard library.

Originally written in 2023 as the programming part of the KUIS AI (Koç University) graduate fellowship application, and refactored in 2026. The original scripts are in [`original/`](original/), and the complexity analysis submitted with them is in [`docs/complexity-analysis.pdf`](docs/complexity-analysis.pdf).

## Method

1. **Tokenise:** keep runs of letters, combining marks and inner apostrophes; drop digits and punctuation; lower-case.
2. **Profile:** pad each token with `_` and count all character n-grams for n = 1, 2, 3 (e.g. `text` → `_t te ex xt t_` for n = 2). Keep the 300 most frequent, in rank order. One profile per language is built from its training text.
3. **Out-of-place distance:** for each n-gram in a document's profile, add the difference between its rank there and its rank in a language profile. An n-gram absent from the language profile costs the profile length.
4. **Identify:** pick the language with the smallest distance.

The working assumption, as in the original method, is that a small set of frequent words and character sequences dominates each language, so their rank order is a stable fingerprint.

## Data

Not included. The original runs used plain-text train/test files drawn from [Universal Dependencies](https://universaldependencies.org/) treebanks for the nine languages. Treebank licences differ (some are CC BY-NC-SA). Any folder works that has one sub-folder per language with `*-train.txt` and `*-test.txt` files:

```
data/
  English/en-ud-train.txt  en-ud-test.txt
  Turkish/tr-ud-train.txt  tr-ud-test.txt
  ...
```

## Usage

```bash
python main.py train    --data data/ --out profiles.json            # --ns 1 2 3 --top-n 300
python main.py identify --profiles profiles.json some_text.txt
python main.py evaluate --profiles profiles.json --data data/
pytest -q
```

No dependencies beyond Python 3.9+ (`pytest` for the tests).

## Results

Profiles trained on the nine `*-train.txt` files (n = 1–3, top 300) and evaluated on the `*-test.txt` files. All nine whole test documents are identified correctly. To measure how much text the method needs, the test files were also cut into non-overlapping snippets of *k* words:

| Snippet length (words) | 1 | 3 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|
| Accuracy | 0.785 | 0.941 | 0.979 | 0.997 | 1.000 | 1.000 |

Almost all errors are among the closely related Germanic languages (English, Dutch, German). Hindi and Urdu, which are close linguistically but use different scripts, are not confused with each other.

## What changed from the 2023 version

- **Tokeniser:** combining marks are now kept inside tokens. The old regex (`\w`) split Hindi and Urdu words at every vowel sign.
- **Settings:** profiles are saved with their n-gram sizes and length. The old test script always built document profiles with n = 1–3 and top 300, whatever the language profiles were trained with.
- **Distance:** it now iterates over the *document* profile, as in Cavnar & Trenkle; the old script iterated over the language profile.
- **Evaluation:** the snippet-length evaluation and unit tests are new.
- **Dependencies:** none; the `tqdm` dependency was dropped.

## Reference

W. B. Cavnar and J. M. Trenkle. *N-Gram-Based Text Categorization.* Proceedings of SDAIR-94, 1994.
