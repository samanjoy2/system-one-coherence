"""Check released records, archive integrity, and common secret patterns offline."""
from pathlib import Path
import json
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = (
    rb"sk-(?:or-v1-)?[A-Za-z0-9_-]{24,}",
    rb"hf_[A-Za-z0-9]{25,}",
    rb"gh[pousr]_[A-Za-z0-9]{30,}",
    rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    rb"Bearer [A-Za-z0-9._-]{24,}",
)


def check_bytes(content, name):
    if any(re.search(pattern, content) for pattern in PATTERNS):
        raise ValueError(f"Potential credential in {name}; contents withheld")


def main():
    assert not (ROOT / ".env").exists(), "Private .env must remain outside a release"
    assert not (ROOT / "paper").exists(), "Manuscript directory found"
    assert not list(ROOT.rglob("*.tex")), "LaTeX files found"
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".py", ".md", ".toml", ".csv"):
            check_bytes(path.read_bytes(), str(path.relative_to(ROOT)))
    counts = {}
    for model, expected_records, expected_raw in (("jev", 3040, 43200), ("laya", 2860, 40800)):
        records = list((ROOT / "results" / model).glob("*/records/*.json"))
        assert len(records) == expected_records, (model, len(records))
        with zipfile.ZipFile(ROOT / "results/cache-archives" / f"{model}-raw.zip") as archive:
            assert len(archive.infolist()) == expected_raw
            for info in archive.infolist():
                content = archive.read(info)  # Also verifies the ZIP CRC.
                check_bytes(content, info.filename)
                row = json.loads(content)
                assert {"example_id", "task", "raw", "probabilities"} <= row.keys()
        counts[model] = {"records": len(records), "raw_questions": expected_raw}
    print(json.dumps(counts, indent=2))
    print("Archive integrity and common-secret-pattern checks passed.")


if __name__ == "__main__":
    main()
