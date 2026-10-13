"""The filter stage as a range-map Mapper."""
from __future__ import annotations

from typing import Optional

from ngramprep.rangemap import Mapper

from ..config import FilterConfig
from ..year_binning import bin_year_records

__all__ = ["FilterMapper", "PivotFilterMapper"]


class FilterMapper(Mapper):
    """Apply :func:`build_processor` to every key and optionally bin the
    years in the value. The FilterConfig is carried as data; the processor
    (and the lemmatizer inside it) is built in each worker."""

    def __init__(self, filter_config: FilterConfig) -> None:
        self.filter_config = filter_config

    def build(self):
        from ..filters.processor_factory import build_processor
        processor = build_processor(self.filter_config)
        bin_size = int(self.filter_config.bin_size or 1)

        def fn(key: bytes, value: bytes) -> Optional[tuple]:
            out = processor(key)
            if not out:
                return None
            if bin_size > 1:
                value = bin_year_records(value, bin_size)
            return out, value

        fn.stats_dict = processor.stats_dict      # per-rule token counts (see processor_factory.STAT_NAMES)
        return fn


class PivotFilterMapper(Mapper):
    """Filter and pivot in one pass: every raw (n-gram, packed years) record
    becomes one record per year, keyed ``4-byte big-endian year + n-gram``
    with the 24-byte (year, count, volumes) triple as value. Raw spellings
    that collapse onto one processed n-gram contribute separate operands for
    the same key, which the packed24 merge operator sums (every operand
    for a key carries the same year, so they merge into one triple); the
    training reader already accepts the 24-byte shape."""

    def __init__(self, filter_config: FilterConfig) -> None:
        self.filter_config = filter_config

    def build(self):
        import struct
        from ..filters.processor_factory import build_processor
        processor = build_processor(self.filter_config)
        bin_size = int(self.filter_config.bin_size or 1)
        pack_year = struct.Struct(">I").pack
        prefixes = {}                           # year -> 4-byte prefix, built as met

        def fn(key: bytes, value: bytes) -> Optional[list]:
            out = processor(key)
            if not out:
                return None
            if bin_size > 1:
                value = bin_year_records(value, bin_size)
            records = []
            for i in range(0, len(value), 24):
                triple = value[i:i + 24]
                year = int.from_bytes(triple[:8], "little")
                prefix = prefixes.get(year)
                if prefix is None:
                    prefix = prefixes[year] = pack_year(year)
                records.append((prefix + out, triple))
            return records

        fn.stats_dict = processor.stats_dict
        return fn
