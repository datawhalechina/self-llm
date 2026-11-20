from pathlib import Path


TEXT_ENCODINGS = ("utf-8", "gb18030", "gbk", "latin-1")


def read_text_with_fallback(file_path):
    """Read a text document using common UTF-8 and Chinese encodings."""
    data = Path(file_path).read_bytes()

    for encoding in TEXT_ENCODINGS:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    raise UnicodeError(f"Could not decode text file: {file_path}")
