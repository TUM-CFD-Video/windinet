#!/usr/bin/env python
"""Fetch natural-video reference clips from the Pexels API (CC0-like Pexels licence) as one tensor.

Output: <out>/clips.pt = uint8 [N, F, 256, 256, 3] plus a JSON with the Pexels ids/urls for attribution.
Only the cropped frames are kept; the mp4s are deleted after decoding.
Key: PEXELS_API_KEY env var, else ~/.config/windinet/pexels_api_key.

    python scripts/wan/download_pexels.py --out results/field_adapter/30_reference/pexels
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import av
import numpy as np
import requests
import torch
import typer

QUERIES = ["nature", "city street", "people walking", "ocean", "forest", "traffic", "animals", "indoor room"]


def _api_key() -> str:
    return (
        os.environ.get("PEXELS_API_KEY") or Path("~/.config/windinet/pexels_api_key").expanduser().read_text().strip()
    )


def _smallest_file(video: dict, min_side: int) -> dict | None:
    ok = [
        f
        for f in video["video_files"]
        if f.get("width") and min(f["width"], f["height"]) >= min_side and f["file_type"] == "video/mp4"
    ]
    return min(ok, key=lambda f: f["width"] * f["height"]) if ok else None


def _middle_frames(path: str, frames: int, size: int) -> np.ndarray | None:
    """`frames` consecutive frames from the clip's middle, centre-cropped and resized so the short side is `size`."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        total = stream.frames or sum(1 for _ in container.decode(stream))
        if total < frames:
            return None
        container.seek(0)
        start, out = (total - frames) // 2, []
        for i, frame in enumerate(container.decode(stream)):
            if i < start:
                continue
            h, w = frame.height, frame.width
            s = min(h, w)
            img = (
                frame.to_image()
                .crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
                .resize((size, size))
            )
            out.append(np.asarray(img))
            if len(out) == frames:
                break
    return np.stack(out) if len(out) == frames else None


def main(
    out: Path = typer.Option(Path("results/field_adapter/30_reference/pexels"), help="output directory (gitignored)"),
    per_query: int = typer.Option(40, help="clips per search term"),
    frames: int = typer.Option(9, help="frames per clip (Wan needs 4k+1)"),
    size: int = typer.Option(256),
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    headers, clips, meta = {"Authorization": _api_key()}, [], []
    for query in QUERIES:
        r = requests.get(
            "https://api.pexels.com/videos/search",
            headers=headers,
            timeout=30,
            params={"query": query, "per_page": per_query, "orientation": "landscape", "size": "small"},
        )
        r.raise_for_status()
        for video in r.json()["videos"]:
            f = _smallest_file(video, size)
            if f is None:
                continue
            with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                tmp.write(requests.get(f["link"], timeout=120).content)
                tmp.flush()
                arr = _middle_frames(tmp.name, frames, size)
            if arr is not None:
                clips.append(arr)
                meta.append({"query": query, "id": video["id"], "url": video["url"], "user": video["user"]["name"]})
        print(f"{query}: {len(clips)} clips so far")
    torch.save(torch.from_numpy(np.stack(clips)), out / "clips.pt")
    (out / "clips.json").write_text(json.dumps(meta, indent=1))
    print(f"saved {len(clips)} clips x {frames} frames to {out}/clips.pt")


if __name__ == "__main__":
    typer.run(main)
