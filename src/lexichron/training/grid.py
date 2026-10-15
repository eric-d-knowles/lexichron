"""The hyperparameter grid: which axes exist, what values each takes, and
the cells (one combination of values each) a grid enumerates.

Settings carry each axis as a list (``vector_size: [100, 300]``); a single
value may be written bare. The UI shows the same lists as rows of toggles
with these presets.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

__all__ = ["AXES", "Axis", "Cell", "Grid", "parse_axis"]


@dataclass(frozen=True)
class Axis:
    name: str                 # settings key and Cell field
    label: str                # as shown
    kind: str                 # "int" or "choice"
    presets: Tuple[Any, ...]  # the toggle row's values (all the values, for a choice)
    default: Tuple[Any, ...]  # lit by default
    help: str
    tag: str                  # short name used in model file stems

    def parse(self, value: Any) -> Tuple[Any, ...]:
        return parse_axis(self, value)


AXES: Tuple[Axis, ...] = (
    Axis("vector_size", "Vector size", "int", (50, 100, 200, 300), (100,),
         "Dimensions of each word vector.", "vs"),
    Axis("window", "Window", "int", (2, 3, 5, 10), (2,),
         "Context words on each side of the target.", "w"),
    Axis("min_count", "Min count", "int", (1, 5, 10, 50), (5,),
         "Words seen fewer times than this in a year's corpus are left out of that year's model.", "mc"),
    Axis("epochs", "Epochs", "int", (5, 10, 20), (5,),
         "Passes over the corpus.", "e"),
    Axis("approach", "Approach", "choice", ("skip-gram", "cbow"), ("skip-gram",),
         "skip-gram predicts context from the word (slower, better for rarer words); cbow the reverse.", "sg"),
    Axis("weight_by", "Weighting", "choice", ("freq", "doc_freq", "none"), ("freq",),
         "How often an n-gram is repeated in the corpus: log of its match count, log of its volume count, or once.",
         "wb"),
)
AXIS_BY_NAME = {a.name: a for a in AXES}


def parse_axis(axis: Axis, value: Any) -> Tuple[Any, ...]:
    """A settings value (list, tuple, bare value or comma-separated string)
    as the axis's tuple of distinct values, validated."""
    if value is None:
        raise ValueError(f"{axis.label}: at least one value is needed")
    if isinstance(value, str):
        items: List[Any] = [v.strip() for v in value.split(",") if v.strip()]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    out: List[Any] = []
    for item in items:
        if axis.kind == "int":
            try:
                v: Any = int(item)
            except (TypeError, ValueError):
                raise ValueError(f"{axis.label}: {item!r} is not a whole number") from None
            if v <= 0:
                raise ValueError(f"{axis.label}: {v} must be positive")
        else:
            v = str(item).strip().lower()
            if v == "skipgram":
                v = "skip-gram"
            if v not in axis.presets:
                raise ValueError(f"{axis.label}: {item!r} is not one of {', '.join(axis.presets)}")
        if v not in out:
            out.append(v)
    if not out:
        raise ValueError(f"{axis.label}: at least one value is needed")
    return tuple(out)


@dataclass(frozen=True)
class Cell:
    vector_size: int
    window: int
    min_count: int
    epochs: int
    approach: str
    weight_by: str

    @property
    def sg(self) -> int:
        return 1 if self.approach == "skip-gram" else 0

    @property
    def stem(self) -> str:
        """The cell's name in file names and tables: ``vs100_w2_mc5_e5_sg1_wbfreq``."""
        return f"vs{self.vector_size}_w{self.window}_mc{self.min_count}_e{self.epochs}_sg{self.sg}_wb{self.weight_by}"

    def as_dict(self) -> Dict[str, Any]:
        return {"vector_size": self.vector_size, "window": self.window, "min_count": self.min_count,
                "epochs": self.epochs, "approach": self.approach, "weight_by": self.weight_by}

    @classmethod
    def from_stem(cls, stem: str) -> "Cell":
        parts = dict(_split_tag(p) for p in stem.split("_"))
        return cls(vector_size=int(parts["vs"]), window=int(parts["w"]), min_count=int(parts["mc"]),
                   epochs=int(parts["e"]), approach="skip-gram" if parts["sg"] == "1" else "cbow",
                   weight_by=parts["wb"])


_TAGS = tuple(sorted((a.tag for a in AXES), key=len, reverse=True))   # longest first: "wb" before "w"


def _split_tag(part: str) -> Tuple[str, str]:
    for tag in _TAGS:
        if part.startswith(tag):
            return tag, part[len(tag):]
    raise ValueError(f"not a cell stem part: {part!r}")


@dataclass(frozen=True)
class Grid:
    values: Dict[str, Tuple[Any, ...]]        # axis name -> values, every axis present

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "Grid":
        """Build from a settings mapping; an axis missing from it takes its default."""
        values = {}
        for axis in AXES:
            raw = settings.get(axis.name)
            values[axis.name] = axis.default if raw is None else axis.parse(raw)
        return cls(values)

    def cells(self) -> List[Cell]:
        names = [a.name for a in AXES]
        return [Cell(**dict(zip(names, combo))) for combo in itertools.product(*(self.values[n] for n in names))]

    def __len__(self) -> int:
        n = 1
        for a in AXES:
            n *= len(self.values[a.name])
        return n

    def describe(self) -> str:
        """``2 vector sizes x 2 windows x ... = 4 cells``."""
        parts = []
        for axis in AXES:
            k = len(self.values[axis.name])
            label = axis.label.lower()
            parts.append(f"{k} {label}{'s' if k != 1 and not label.endswith('s') else ''}"
                         if k != 1 else f"1 {label}")
        return " x ".join(parts) + f" = {len(self)} cell{'s' if len(self) != 1 else ''}"

    def as_settings(self) -> Dict[str, List[Any]]:
        return {a.name: list(self.values[a.name]) for a in AXES}


def corpus_variants(cells: Iterable[Cell]) -> List[str]:
    """The weightings a set of cells needs corpus files for (one file per
    year and weighting; the other axes only change training)."""
    seen: List[str] = []
    for c in cells:
        if c.weight_by not in seen:
            seen.append(c.weight_by)
    return seen
