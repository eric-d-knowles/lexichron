# How the stages work inside

The README says what each stage does and what it guarantees. This file says
how, for anyone changing the code or checking a result.

## Acquisition

A run is one parent process and a pool of worker processes, and only the
parent writes to the database:

1. **The parent lists the files.** It reads the release's index page for the
   corpus and n-gram size, keeps the `file_range` if one is set, and drops
   every file already marked as processed in the database (`overwrite_db`
   deletes the database first instead; the command line and the UI ask
   before it does). `random_seed` shuffles the order so a partial run
   samples the alphabet evenly.
2. **The parent hands files to the workers.** Each worker gets one URL at a
   time, with at most two files per worker queued ahead.
3. **A worker downloads and parses its file in one stream.** It decompresses
   the gzip as it arrives, parses each line, applies the `ngram_type` filter
   and any `combined_multigrams`, and packs the year, count and volume
   triplets into binary. It holds at most `chunk_entries` entries in memory,
   whatever the file's size: each full batch is sorted and written under
   `spool_dir` as an SST file (RocksDB's own file format), repeated keys
   within it summed, and at the end of the file the batches are merged into
   one sorted SST file. A connection dropped mid-file, or a body shorter
   than the server announced, discards what was written and starts the file
   again, up to three times.
4. **The parent adopts each finished file.** It hands the worker's SST file
   to the database together with a second, one-key file holding the file's
   "processed" marker, in a single ingest call. RocksDB applies the call as
   one edit of its manifest, so a file's records and its marker are in the
   database together or not at all, wherever the job is killed: an
   interrupted run loses only the files in flight, and the next run
   (`overwrite_db: false`) picks up the rest. Nothing is replayed record by
   record, so the parent keeps up with any number of workers.
5. **The parent compacts the database.** Each file arrived as its own SST
   file and auto-compaction is off in the write profile, so this is what
   folds them together. It is not optional: until then an n-gram that
   arrived from several files exists as several partial records. The one
   case in which it is skipped is a re-run that adds no file to a database
   a previous run already compacted (a job that ended during the count
   check, say): every SST file is then in one sorted level, no n-gram can
   be held twice, and compacting again would rewrite the whole database to
   produce the same records.
6. **The counts are verified** (`verify_counts`, on by default; see below).
7. **The parent archives the database** when `archive_path_stub` is set,
   as a `.tar.zst` under `<stub>/<release>/<corpus>/<n>gram_files/`.

A file that fails all its attempts does not stop the others. The run reports
it at the end and exits with an error, skipping compaction, verification and
archiving; a rerun resumes with only the missing files.

**Counts are summed, never replaced.** Every record is written as a *merge*:
when a key is already in the database, its per-year counts are added. Without
`combined_multigrams` each n-gram occurs once in the whole release and the
sum has one term. With them it does not: combining keeps only the first
token's tag, so `working_NOUN class_NOUN` and `working_NOUN class_VERB` both
become `working-class_NOUN`, and a sequence with a named token collapses its
capitalizations too (`a . i .` and `A . I .` both become `A.I.`), which sit
in different files. Their counts are added up within a batch by the worker,
across a file's batches when they are merged, and across files by the
compaction.

Summing has one hazard, which is why the run checks its own arithmetic: a
file taken in twice would be counted twice, and no later stage could tell.
Four things guard the counts.

- **A ledger, checked against the database on every run.** Each worker
  totals the occurrences and the volume counts of every line it keeps, and
  the totals are stored in the file's marker. Adding counts together never
  changes a grand total, so after compaction the database must hold exactly
  the sum of its markers' totals. The run reads the whole database once, in
  parallel, and compares: a file ingested twice, a record lost on the way or
  a wrong sum each leave a difference, the run fails, and the summary gives
  both figures. The outcome is recorded in the database's provenance
  (`lexichron inspect <database> --info`) and shown in the UI. A database
  begun by an earlier version, whose markers hold no totals, is reported as
  not verifiable; that is not an error.
- **Years in order.** The merge operator assumes each value's years ascend
  and checks nothing. Workers check every value before it is spooled and
  rewrite any that arrive otherwise, and the verification scan checks every
  value again in the finished database.
- **No line dropped in silence.** A line whose counts cannot be read in full
  (an entry that is not `year,count,volumes` in whole numbers) is logged
  with its content and counted in the file's marker; what can be read of it
  is kept. A run that met any such line finishes its work and then fails,
  saying how many, because those n-grams are undercounted.
  `allow_unparsed_lines: true` accepts them; the number is recorded either
  way.
- **Records and marker together.** Step 4 above. `tests/test_acquire_kill.py`
  kills a run at chosen and at random moments against the real RocksDB and
  requires the resumed database to equal an uninterrupted one.

## Filtering

The stage is built like the acquire stage, with a database where that has
downloads. The work is split into key ranges (*work units*) balanced by data
volume, from the source database's own file metadata, so there is no
pre-scan. Worker processes each scan a unit, and hand it over as one sorted
SST file; the parent adopts that file together with the unit's marker in a
single ingest call, so a unit's records and the fact that it is done are in
the destination together or not at all, wherever the job is killed. When
every unit is in, the parent compacts the destination, which is what folds
the units' files into one sorted database and sums the counts of records
that arrived from more than one unit. Compaction is not optional. A worker
holds the source open with rocks-shim's `scan` profile (only the files its
range touches, nothing pinned, a small cache), so its memory does not grow
with the database: about 280 MB while it scans, and more for a moment at the
end of each unit, when it merges the sorted parts it spooled into one file.
For a pivoted run that is a thousand or two parts and a peak of 0.5 to
0.85 GB (measured on the English 5-grams); the Submit tab's estimate allows
0.9 GB a worker plus 4 GB, so 40 workers fit in the default 40G, and 0.4 GB
a worker for a unigram run. Slurm's own `MaxRSS` will read much higher than
that for the same job, because it counts the pages forked workers share once
per worker, and a job's memory also fills with page cache from the spool
files, which is given back under pressure. A run that is interrupted resumes
from the units it had finished (`mode: resume`); `mode: restart` starts over,
after deleting the destination database, and both the UI and the command
line ask before it does (see [Command line](#command-line)).
A resume finishes only the units not yet done, so it must be given what the
run that started the destination was given: the destination records the
vocabulary (by its content, not its path), how lemmas are made, the keep
lists and the rules, and a resume with any of them changed is refused with
the difference named. A rebuilt vocabulary therefore needs `mode: restart`;
resuming would have finished the database with units filtered two ways, and
every count check would have passed.

The source must be a finished database: one whose acquisition stopped before
compacting (a file failed, the job ran out of time) is refused, with a
message saying to run that stage again.

The count check keeps three totals, each recorded in every unit's marker.
Filtering only moves counts and adds them together, so they must fit
together exactly:

- **What was read.** Every unit totals the occurrences and volume counts of
  every source record in its range, kept or not. Each record lies in exactly
  one unit, so the units together must have read exactly what the source
  holds. Where the acquisition that made the source verified its counts,
  that figure is on record and the two are compared: a key range the
  partition skipped, or gave to two units, shows as a difference. A source
  acquired before counts were verified has no figure to compare with, and
  the run says so.
- **What was written.** Every unit totals what it emits, before records with
  the same key are summed. After compaction the destination is read once, in
  parallel, and must hold exactly the sum of what the units emitted, with
  every value's years in ascending order (workers check that as they spool,
  and put right any value that is not). A unit taken in twice, or a record
  lost on the way, shows as a difference.
- **What was dropped.** Every unit also totals the n-grams it drops, from
  those records themselves and not by subtraction. A kept n-gram goes out
  with every count it came in with (lowercased, lemmatized, binned, split
  into one record per year, but not a count more or less), so what was read
  must equal what was kept plus what was dropped. The two checks above
  cannot see a count lost in between, in binning or in the pivot: the
  source would still be read in full and the destination would still hold
  what was emitted. This one can. It also gives the share of occurrences
  the filter removed, shown on the Progress tab as the run goes.

A Davies corpus is acquired and filtered by the same machinery: its worker
reads one archive and hands over one sorted SST file with its marker, as an
n-gram worker does with one file.

## The tuning scheduler

`training/scheduler.py` trains the models of all the sampled years as one
queue in year order (train uses it too, keyed by year and replicate): a
training slot that frees up takes the next
model whose corpus is ready, and corpora are written one year ahead, so
a year's last models and the next year's first ones train side by side
(a year's corpus and non-seed-0 models are deleted when its last model
lands). Within a year the costliest cells are queued first (by epochs ×
vector size, × the window for skip-gram), so the short ones fill the
slots as the long ones free them; a cell's models stay together and it
is scored when its last one lands. A model's threads are fixed when it
starts: one that starts when fewer models remain than there are slots
gets the spare CPUs (`workers_per_model` is a minimum). The long models
start first, with `workers_per_model` threads, so a task runs about as
long as its costliest model takes and uses fewer cores toward the end: a
search over 100 to 300 dimensions and 10 to 30 epochs used 55 to 62% of
its CPU time. The image rebuilds gensim's `corpus_file` reader free
of the GIL (`scripts/patch_gensim.py`): the stock build re-takes the GIL
once per line, which on five-token n-gram lines caps a model at three or
four cores however many threads it has.
