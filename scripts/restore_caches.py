"""Restore native-response archives without network access."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def restore():
    for model in ("jev", "laya"):
        count = 0
        with zipfile.ZipFile(ROOT / "results/cache-archives" / f"{model}-raw.zip") as archive:
            for info in archive.infolist():
                parts = Path(info.filename).parts
                if (len(parts) != 5 or parts[:2] != ("results", model)
                        or parts[3] != "raw" or not parts[4].endswith(".json")):
                    raise ValueError("Unexpected archive member")
                target = (ROOT / info.filename).resolve()
                if not target.is_relative_to(ROOT / "results" / model):
                    raise ValueError("Archive member escapes model directory")
                content = archive.read(info)
                if target.exists():
                    if target.read_bytes() != content:
                        raise ValueError(f"Existing cache differs: {info.filename}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as stream:
                    stream.write(content)
                count += 1
        print(f"{model}: restored {count} question records")


if __name__ == "__main__":
    restore()
