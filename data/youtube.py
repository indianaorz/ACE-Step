#!/usr/bin/env python3
"""
yt_dl_split_mp3.py
──────────────────
Download one YouTube URL, convert best audio to a single MP3, then
split that MP3 into chapter files using yt‑dlp’s chapter metadata.

Requires:  yt‑dlp ≥ 2021‑10‑09,  FFmpeg in your PATH
"""

from __future__ import annotations
import subprocess, sys, re
from pathlib import Path
import yt_dlp


# ─────────────── helpers ────────────────
def slugify(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z._ -]", "", text).strip().replace("  ", " ")

def ffmpeg_slice(src: Path, dst: Path, start: float, end: float | None) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
           "-y", "-i", str(src), "-vn", "-acodec", "copy",
           "-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += [str(dst)]
    subprocess.run(cmd, check=True)


# ─────────────── main workflow ────────────────
def main(url: str) -> None:
    ydl_cfg = {
        "quiet": True,
        "format": "bestaudio/best",
        "outtmpl": {"default": "%(title)s.%(ext)s"},  # single file
    }

    with yt_dlp.YoutubeDL(ydl_cfg) as ydl:
        dl_info = ydl.extract_info(url, download=True)  # download & return dict

    title = slugify(dl_info["title"])

    # ── Locate the downloaded file ───────────────────────────────────────
    try:
        # yt‑dlp ≥ 2023 appends real path(s) here
        audio_src = Path(dl_info["requested_downloads"][0]["filepath"])
    except (KeyError, IndexError):
        # Fallback: yt‑dlp’s filename generator
        audio_src = Path(yt_dlp.YoutubeDL().prepare_filename(dl_info))

    if not audio_src.exists():
        raise FileNotFoundError(
            f"Could not locate downloaded audio file at {audio_src!s}"
        )
    # ─────────────────────────────────────────────────────────────────────

    audio_mp3 = audio_src.with_suffix(".mp3")

    print("🎧  Transcoding full audio → MP3 …")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio_src),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "2",  # ≈192 kbps VBR
            str(audio_mp3),
        ],
        check=True,
    )
    audio_src.unlink()  # remove original container

    chapters = dl_info.get("chapters") or []
    if not chapters:
        print("⚠️  No chapter data found – nothing to split.")
        return

    out_dir = Path(title)
    out_dir.mkdir(exist_ok=True)

    print("✂️   Splitting chapters …")
    for idx, ch in enumerate(chapters, 1):
        part = out_dir / f"{idx:02d} {slugify(ch['title'])}.mp3"
        ffmpeg_slice(audio_mp3, part, ch["start_time"], ch.get("end_time"))
        print(f"   • {part.name}")

    print("🗑️  Cleaning up full MP3 …")
    audio_mp3.unlink()

    print("✅  Done!  Chapters saved in:", out_dir.resolve())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:  yt_dl_split_mp3.py <YouTube URL>")
        sys.exit(1)
    main(sys.argv[1])
