"""Verify the evaluation datasets baked into the image and write SOURCES.md.

Usage: check_datasets.py DIR

The files come from gensim's repository (its test data), which carries the
canonical copies of SimLex-999 (Hill, Reichart & Korhonen 2015), WordSim-353
(Finkelstein et al. 2002) and Google's analogy questions (Mikolov et al.
2013). Each is checked for shape (line counts and record format) so that a
truncated or substituted download fails the build; its SHA-256 is recorded
next to it for the provenance records of runs that use it.
"""
import hashlib
import sys
from pathlib import Path

EXPECTED = {
    # file: (min pairs/questions, a check on one record)
    "simlex999.txt": (999, "similarity"),
    "wordsim353.tsv": (353, "similarity"),
    "questions-words.txt": (19000, "analogy"),
    "vader_lexicon.txt": (7000, "ratings"),
}
SOURCES = {
    "simlex999.txt": "SimLex-999: Hill, Reichart & Korhonen (2015), Computational Linguistics 41(4).",
    "wordsim353.tsv": "WordSim-353: Finkelstein et al. (2002), ACM TOIS 20(1).",
    "questions-words.txt": "Google analogy test set: Mikolov et al. (2013), 'Efficient Estimation of Word Representations in Vector Space'.",
    "vader_lexicon.txt": "VADER sentiment lexicon (human valence ratings, -4..4): Hutto & Gilbert (2014), ICWSM-14; MIT licence, "
                         "from https://github.com/cjhutto/vaderSentiment.",
}


def count_records(path: Path, kind: str) -> int:
    n = 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(":"):
                continue
            parts = line.replace(",", "\t").split("\t") if kind == "similarity" and ("\t" in line or "," in line) else line.split()
            if kind == "similarity":
                # the score is the first numeric field after the two words
                # (SimLex-999 has a part-of-speech column before it)
                if len(parts) >= 3 and any(_is_number(f) for f in parts[2:]):
                    n += 1
            elif kind == "ratings":
                # a plain word and a numeric rating (VADER: token, mean, sd, ratings)
                parts = line.split("\t")
                if len(parts) >= 2 and parts[0].isalpha() and _is_number(parts[1]):
                    n += 1
            elif len(parts) == 4:
                n += 1
    return n


def _is_number(f: str) -> bool:
    try:
        float(f)
        return True
    except ValueError:
        return False


def main(argv):
    d = Path(argv[1])
    lines = ["# Evaluation datasets", "",
             "Fetched at image build from https://github.com/piskvorky/gensim (gensim/test/test_data) and "
             "https://github.com/cjhutto/vaderSentiment (vaderSentiment/vader_lexicon.txt).", ""]
    ok = True
    for name, (minimum, kind) in EXPECTED.items():
        p = d / name
        if not p.is_file():
            print(f"missing: {p}", file=sys.stderr)
            ok = False
            continue
        n = count_records(p, kind)
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        if n < minimum:
            print(f"{name}: only {n} records (expected at least {minimum})", file=sys.stderr)
            ok = False
        lines.append(f"- `{name}`: {n} records, sha256 {sha}  \n  {SOURCES[name]}")
        print(f"{name}: {n} records, sha256 {sha[:16]}...")
    (d / "SOURCES.md").write_text("\n".join(lines) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
