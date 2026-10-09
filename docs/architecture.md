# Pipeline architecture

The `ngram_filter` and `ngram_pivot` pipelines use a two-phase design for memory efficiency and fault tolerance:

1. **Processing stage**: Workers divide the input data into chunks, process them in parallel, and write results to temporary files (`tmp_dir/worker_outputs/`)
2. **Ingestion stage**: Temporary files are merged into the final database using parallel streaming

This design enables:

- **Resume capability**: Interrupted jobs pick up where they left off
- **Load balancing**: Work units are pre-balanced via density-based sampling; workers steal remaining units as they finish
- **Balanced work units**: Density-based sampling scans the corpus to estimate token frequency distributions, then partitions work so each unit has similar total token mass, reducing straggler workers and keeping throughput consistent
- **Memory efficiency**: Large datasets don't need to fit in RAM
- **Predictable resource usage**: Memory consumption is bounded regardless of corpus size

*Note: Davies acquisition pipelines use simpler direct ingestion and do not employ the two-stage architecture.*

## Acquisition: streaming through chunks

`ngram_acquire` workers do not hold a whole shard in memory. Each worker
parses its shard and spools packed entries to chunk files (default 200,000
entries each) under the spool directory, keeping one chunk in memory. Chunks
are written as `.part` files and renamed to `.chunk` only once the whole shard
has parsed, so a shard retried after a dropped connection cannot be counted
twice. The parent process then streams each completed shard's chunks into
RocksDB with `merge` (keys repeated across chunks are summed by the packed24
merge operator), writes the shard's resume marker, and flushes. Peak memory is
about one chunk per worker plus one in the parent, plus RocksDB's own
memtables, independent of shard size.
