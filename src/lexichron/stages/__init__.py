"""Plain-value entry points for pipeline stages.

The Python APIs in ngramprep take objects (a stopword set, a lemmatizer).
These wrappers take only values that fit in a settings file (strings,
numbers, booleans, lists) and build the objects, so the same function serves
the command line, the terminal UI and a notebook.
"""
