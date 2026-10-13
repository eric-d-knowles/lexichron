"""The filter stage as a range-map Mapper."""
from __future__ import annotations

from typing import Optional

from ngramprep.rangemap import Mapper

from ..config import FilterConfig
from ..year_binning import bin_year_records

__all__ = ["FilterMapper"]


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
