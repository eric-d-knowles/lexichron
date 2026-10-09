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
