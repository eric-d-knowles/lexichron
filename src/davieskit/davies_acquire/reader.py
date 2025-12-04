"""Text file reading and parsing for Davies corpora."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Iterator, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "discover_text_files",
    "extract_year_from_filename",
    "read_text_file",
]


def discover_text_files(text_dir: Path) -> list[Path]:
    """
    Discover all text archive files in the text directory.

    Args:
        text_dir: Directory containing text zip files

    Returns:
        Sorted list of text file paths

    Example:
        >>> files = discover_text_files(Path("/data/COHA/text"))
        >>> files[0]
        Path('/data/COHA/text/text_1810s_kso.zip')
    """
    if not text_dir.exists():
        raise ValueError(f"Text directory does not exist: {text_dir}")

    # Find all .zip files matching text_*
    pattern = "text_*.zip"
    files = sorted(text_dir.glob(pattern))

    if not files:
        raise ValueError(f"No text files found in {text_dir}")

    logger.info(f"Found {len(files)} text archive files")
    return files


def extract_year_from_filename(
    filename: str,
    decade_pattern: str = r"text_(\d{4})s_",
    year_pattern: Optional[str] = None,
) -> int:
    """
    Extract year/decade from Davies corpus filename.

    For decade-based corpora (like COHA), extracts the starting year of the decade.
    For year-based corpora, extracts the specific year.

    Args:
        filename: Filename to parse (e.g., "text_1810s_kso.zip")
        decade_pattern: Regex pattern for decade extraction
        year_pattern: Regex pattern for year extraction (optional)

    Returns:
        Year as integer (e.g., 1810 for "text_1810s_kso.zip")

    Raises:
        ValueError: If no year/decade found in filename

    Example:
        >>> extract_year_from_filename("text_1810s_kso.zip")
        1810
    """
    # Try decade pattern first
    if decade_pattern:
        match = re.search(decade_pattern, filename)
        if match:
            return int(match.group(1))

    # Try year pattern if provided
    if year_pattern:
        match = re.search(year_pattern, filename)
        if match:
            return int(match.group(1))

    raise ValueError(f"Could not extract year from filename: {filename}")


def read_text_file(
    zip_path: Path,
    year: int,
) -> Iterator[Tuple[int, str]]:
    """
    Read and yield document text from a Davies corpus zip file.

    Each zip contains multiple text documents. This function yields
    the text content of each document along with its year.

    Args:
        zip_path: Path to zip file
        year: Year associated with this file

    Yields:
        Tuples of (year, document_text)

    Example:
        >>> for year, text in read_text_file(Path("text_1810s.zip"), 1810):
        ...     print(f"Year {year}: {len(text)} chars")
    """
    logger.debug(f"Reading {zip_path.name} (year={year})")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # Get all .txt files in the archive
            txt_files = [f for f in zf.namelist() if f.endswith('.txt')]

            if not txt_files:
                logger.warning(f"No .txt files found in {zip_path.name}")
                return

            for txt_file in txt_files:
                try:
                    # Read file content
                    with zf.open(txt_file) as f:
                        content = f.read().decode('utf-8', errors='replace')

                    # Skip document ID markers (e.g., "@@552651")
                    # These appear at the start of COHA text files
                    if content.startswith('@@'):
                        # Remove the marker line
                        lines = content.split('\n', 1)
                        if len(lines) > 1:
                            content = lines[1]

                    # Only yield if there's actual content
                    if content.strip():
                        yield year, content

                except Exception as e:
                    logger.warning(f"Error reading {txt_file} from {zip_path.name}: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error opening {zip_path}: {e}")
        raise
