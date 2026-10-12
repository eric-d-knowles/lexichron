"""Range-partitioned map over a RocksDB with a single merging writer.

The engine behind the filter stages. It applies the architecture that the
acquisition stage uses, to a RocksDB source instead of downloaded files:

* the key space is cut into many *units* (key ranges), balanced by sampling;
* worker processes each scan one unit read-only, apply a *mapper* to every
  record, and spool the results to bounded chunk files (never holding more
  than one chunk in memory, and never opening a writable database);
* the parent process merges each finished unit's chunks into the destination
  database, flushes, and writes a marker for the unit, so a crash loses at
  most the unit in progress and a re-run resumes from the markers;
* work is handed out by the parent, so there is nothing for workers to
  contend over.

See :func:`run_rangemap` and :class:`Mapper`.
"""
from .engine import Mapper, RangeMapConfig, RangeMapResult, run_rangemap
from .units import PARTITION_KEY, UNIT_MARKER_PREFIX

__all__ = ["Mapper", "RangeMapConfig", "RangeMapResult", "run_rangemap",
           "PARTITION_KEY", "UNIT_MARKER_PREFIX"]
