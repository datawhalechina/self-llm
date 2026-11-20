import tempfile
from pathlib import Path

from encoding_utils import read_text_with_fallback


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "知识库.txt"
        expected = "价格说明\n仅支持中文文档。"
        path.write_bytes(expected.encode("gb18030"))

        text, encoding = read_text_with_fallback(path)

        assert text == expected
        assert encoding == "gb18030"
        print(f"decoded {path.name} as {encoding}: {text!r}")


if __name__ == "__main__":
    main()
