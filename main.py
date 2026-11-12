"""N-gram language identification (Cavnar & Trenkle, 1994).

    python main.py train    --data DATA_DIR --out profiles.json [--ns 1 2 3] [--top-n 300]
    python main.py identify --profiles profiles.json FILE [FILE ...]
    python main.py evaluate --profiles profiles.json --data DATA_DIR [--lengths 1 3 5 10 20 50]

DATA_DIR has one sub-folder per language containing *-train.txt and
*-test.txt files (the folder name is the language label).
"""
import argparse
import glob
import os

from langid.evaluate import evaluate
from langid.profiles import LanguageIdentifier


def read_split(data_dir, split):
    texts = {}
    for lang_dir in sorted(glob.glob(os.path.join(data_dir, "*", ""))):
        files = sorted(glob.glob(os.path.join(lang_dir, f"*-{split}.txt")))
        if files:
            texts[os.path.basename(os.path.normpath(lang_dir))] = "\n".join(
                open(f, encoding="utf-8").read() for f in files)
    if not texts:
        raise SystemExit(f"no *-{split}.txt files found under {data_dir}")
    return texts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--data", required=True)
    tr.add_argument("--out", required=True)
    tr.add_argument("--ns", type=int, nargs="+", default=[1, 2, 3])
    tr.add_argument("--top-n", type=int, default=300)
    idf = sub.add_parser("identify")
    idf.add_argument("--profiles", required=True)
    idf.add_argument("files", nargs="+")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--profiles", required=True)
    ev.add_argument("--data", required=True)
    ev.add_argument("--lengths", type=int, nargs="+", default=[1, 3, 5, 10, 20, 50])
    args = parser.parse_args()

    if args.command == "train":
        model = LanguageIdentifier.train(read_split(args.data, "train"), args.ns, args.top_n)
        model.save(args.out)
        print(f"saved {len(model.profiles)} profiles (n={model.ns}, top {model.top_n}) to {args.out}")
    elif args.command == "identify":
        model = LanguageIdentifier.load(args.profiles)
        for path in args.files:
            print(f"{path}\t{model.identify(open(path, encoding='utf-8').read())}")
    else:
        model = LanguageIdentifier.load(args.profiles)
        test = read_split(args.data, "test")
        print("whole test files: " + ", ".join(
            f"{lang}->{model.identify(text)}" for lang, text in test.items()))
        for n, r in evaluate(model, test, args.lengths).items():
            worst = ", ".join(f"{a}->{b} ({c})" for (a, b), c in r["confusions"].most_common(3))
            print(f"{n:>3}-word snippets: accuracy {r['accuracy']:.3f}   most common errors: {worst or '-'}")


if __name__ == "__main__":
    main()
