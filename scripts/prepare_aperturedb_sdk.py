"""Prepare the local uv index used by the ApertureDB integrations."""

from pathlib import Path
import re
from subprocess import run
from tempfile import TemporaryDirectory

image = "registry-1.docker.io/aperturedata/aperturedb-python:latest"
index = Path(__file__).resolve().parents[1] / ".aperturedb-sdk"
index.mkdir(exist_ok=True)
with TemporaryDirectory(dir=index) as directory:
    staging = Path(directory)
    run(["oras", "pull", "--output", directory, image], check=True)
    wheel, = staging.glob("aperturedb-*-py3-none-any.whl")
    if not re.fullmatch(r"aperturedb-\d+\.\d+\.\d+-py3-none-any\.whl", wheel.name):
        raise ValueError(f"Unexpected SDK wheel name {wheel.name!r}")
    target = wheel.replace(index / wheel.name)
    page = staging / "index.html"
    page.write_text(f'<a href="{target.name}">{target.name}</a>\n', encoding="utf-8")
    page.replace(index / "index.html")
print(target)
