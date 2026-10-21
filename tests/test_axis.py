"""The valence axis as a yardstick: validity against ratings, agreement
between models, and the file readers."""
import numpy as np

from lexichron.training import axis
from lexichron.training.trainer import Vectors


def _planted(seed: int, noise: float = 0.2, dim: int = 16):
    """Vectors where a hidden direction carries each word's valence: pole words
    at the ends, rated words along it in proportion to their rating."""
    rng = np.random.default_rng(seed)
    pleasant = [f"good{i}" for i in range(5)]
    unpleasant = [f"bad{i}" for i in range(5)]
    targets = {f"t{i:02d}": (i - 15) / 5 for i in range(30)}          # ratings -3..+2.8
    d = np.zeros(dim); d[0] = 1.0
    words, rows = [], []
    for w in pleasant:
        words.append(w); rows.append(3 * d + noise * rng.normal(size=dim))
    for w in unpleasant:
        words.append(w); rows.append(-3 * d + noise * rng.normal(size=dim))
    for w, r in targets.items():
        words.append(w); rows.append(r * d + rng.normal(size=dim) * noise + 0.5 * rng.normal(size=dim) * np.eye(dim)[1])
    return Vectors(words=words, matrix=np.array(rows, dtype=np.float32), counts=np.arange(len(words), 0, -1)), \
        axis.Axis(pleasant, unpleasant, targets)


def test_validity_recovers_the_planted_ratings():
    v, yard = _planted(1)
    rho, cov = axis.validity(v, yard)
    assert rho > 0.9 and cov == 1.0
    # the poles are kept out of the rated targets
    assert not any(w in yard.ratings for w in yard.pleasant + yard.unpleasant)


def test_agreement_is_high_between_near_copies_and_lower_with_noise():
    a, yard = _planted(1)
    b, _ = _planted(2)
    c, _ = _planted(3, noise=1.5)
    assert axis.agreement(a, b, yard) > 0.9
    assert axis.agreement(a, c, yard) < axis.agreement(a, b, yard)


def test_nan_without_poles_or_enough_targets():
    v, yard = _planted(1)
    few = axis.Axis(yard.pleasant, yard.unpleasant, {w: r for w, r in list(yard.ratings.items())[:5]})
    assert np.isnan(axis.validity(v, few)[0]) and np.isnan(axis.agreement(v, v, few))
    nopoles = axis.Axis(["absent1", "absent2", "absent3"], yard.unpleasant, yard.ratings)
    assert axis.axis_vector(v, nopoles.pleasant, nopoles.unpleasant) is None
    assert np.isnan(axis.validity(v, nopoles)[0])


def test_readers(tmp_path):
    lex = tmp_path / "vader_lexicon.txt"
    lex.write_text("$:\t-1.5\t0.8\t[-1, -1]\n%)\t-0.4\t1.0\t[0, 0]\nlove\t3.2\t0.4\t[3, 3]\nhate\t-2.7\t1.0\t[-3, -2]\n"
                   "Word\tV.Mean.Sum\nwell-being\t2.0\t0.1\t[2]\n")
    assert axis.read_ratings(lex) == {"love": 3.2, "hate": -2.7}           # plain words only; header skipped
    csvf = tmp_path / "warriner.csv"
    csvf.write_text("Word,V.Mean.Sum,V.SD.Sum\naardvark,6.26,2.21\nabandon,2.84,1.54\n")
    assert axis.read_ratings(csvf) == {"aardvark": 6.26, "abandon": 2.84}
    poles = tmp_path / "poles.txt"
    poles.write_text("# pleasant, then unpleasant\nlove, joy peace\nfear war\n")
    assert axis.read_poles(poles) == (["love", "joy", "peace"], ["fear", "war"])
    yard = axis.load_axis(str(lex), str(poles))
    assert yard.pleasant == ("love", "joy", "peace") and yard.ratings == {"hate": -2.7}   # love is a pole now
    assert axis.load_axis(None, None) is None
    default = axis.load_axis(str(lex), None)
    assert default.pleasant == axis.PLEASANT and len(axis.PLEASANT) == len(axis.UNPLEASANT) == 25
