"""lexichron command-line interface and project configuration."""

__version__ = "0.3.0"

# Images built by hand from a branch (the `dev`, `filter`, ... tags) carry a
# build stamp written by the container workflow into lexichron/_build.py, so
# that `lexichron --version`, the UI title and the installed image's file
# name say which build this is (0.3.0+dev.79ad8ed). Release images have
# no stamp and report the plain version.
try:
    from lexichron._build import BUILD as __build__
except ImportError:  # a release, or a source checkout
    __build__ = ""
if __build__:
    __version__ = f"{__version__}+{__build__}"
