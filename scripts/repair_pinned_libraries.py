"""Restore the two project-local pinned wheels after macOS file eviction."""
from __future__ import annotations

import hashlib
import io
import json
import os
import platform
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT/".venv/lib/python3.12/site-packages"
CACHE = Path("/private/tmp/info5600_pinned_wheels")


def main():
    assert sys.platform == "darwin" and platform.machine() == "arm64" and sys.version_info[:2] == (3,12), "This local recovery targets macOS arm64 with Python 3.12."
    CACHE.mkdir(parents=True,exist_ok=True)
    records = []
    for package,version,prefixes in [("interpret-core","0.7.8",["interpret/","interpret_core-0.7.8.dist-info/"]),
                                     ("lightgbm","4.6.0",["lightgbm/","lightgbm-4.6.0.dist-info/"])]:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/{version}/json",timeout=30) as response:
            metadata = json.load(response)
        choices = [entry for entry in metadata["urls"] if entry["filename"].endswith(".whl")]
        if package == "interpret-core":
            wheel = next(entry for entry in choices if entry["filename"].endswith("py3-none-any.whl"))
        else:
            wheel = next(entry for entry in choices if "macosx" in entry["filename"] and "arm64" in entry["filename"])
        assert wheel["url"].startswith("https://files.pythonhosted.org/")
        cached = CACHE/wheel["filename"]
        if not cached.exists():
            with urllib.request.urlopen(wheel["url"],timeout=60) as response:body = response.read()
            cached.write_bytes(body)
        body = cached.read_bytes()
        checksum = hashlib.sha256(body).hexdigest()
        assert checksum == wheel["digests"]["sha256"]
        count = 0
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            for entry in archive.infolist():
                if entry.is_dir() or not any(entry.filename.startswith(prefix) for prefix in prefixes):continue
                destination = TARGET/entry.filename
                assert destination.resolve().is_relative_to(TARGET.resolve())
                destination.parent.mkdir(parents=True,exist_ok=True)
                temporary = destination.with_name(destination.name+".restored")
                temporary.write_bytes(archive.read(entry))
                os.replace(temporary,destination)
                count += 1
        record = {"package":package,"version":version,"wheel":wheel["filename"],"sha256":checksum,
                  "sha256_verified_against_pypi":True,"files_restored":count,"model_parameters_changed":False}
        records.append(record)
        print(json.dumps(record),flush=True)
    output = ROOT/"outputs/feature_optimization/dependency_recovery.json"
    output.write_text(json.dumps({"restored":records},indent=2)+"\n")


if __name__ == "__main__":main()
