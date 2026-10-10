import os
import sys
import subprocess
from pathlib import Path

# Enchant library path fix
env_path = Path(sys.executable).parent.parent
os.environ.setdefault('PYENCHANT_LIBRARY_PATH', str(env_path / 'lib' / 'libenchant-2.so'))

# Fix hunspell dictionary path if needed
hunspell_src = env_path / 'share' / 'hunspell_dictionaries'
hunspell_dst = env_path / 'share' / 'hunspell'
if hunspell_src.exists() and not hunspell_dst.exists():
    try:
        hunspell_dst.symlink_to(hunspell_src)
    except FileExistsError:
        pass

# Download spaCy models if missing
SPACY_MODELS = [
    'en_core_web_sm',
    'zh_core_web_sm',
    'fr_core_news_sm',
    'de_core_news_sm',
    'it_core_news_sm',
    'ru_core_news_sm',
    'es_core_news_sm',
]

def _ensure_spacy_models():
    import spacy
    for model in SPACY_MODELS:
        try:
            spacy.load(model)
        except OSError:
            print(f"Downloading {model}...")
            subprocess.run([sys.executable, '-m', 'spacy', 'download', model], check=True)

_ensure_spacy_models()

"""ngramprep: N-gram data preparation tools."""
__version__ = "0.2.0"
__all__ = [
    "__version__",
]