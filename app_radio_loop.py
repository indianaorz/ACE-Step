#!/usr/bin/env python3
"""
LoRA Radio  v11.5 – Full Page BG Spectrogram, Playhead, Y-Axis Window
  • Streams only the LATEST FLAC whose filename starts with "pred_"
  • Loops that latest song WITH SERVER-SIDE CROSSFADE.
  • When a new song is added, browser is notified via SSE.
  • Client fades out current song, then fades in new song.
  • Client-side low-pass and high-pass filters.
  • UI: Listen/Stop buttons, filter controls, live spectrogram.
  • Full page layout.
  • REAL Filtered full-song spectrogram display as dynamic page background.
  • Playhead on the full-song spectrogram.
  • Corrected color mapping for spectrograms.
  • Slider for Y-axis (frequency) window on background spectrogram.
  MODIFIED: Added Y-Axis frequency window controls for the background spectrogram.
"""

from __future__ import annotations

import argparse, logging, struct, time
from pathlib import Path
from threading import Event, Lock, Thread
import queue # Explicit import for queue.Empty

import numpy as np
import soundfile as sf  # pip install soundfile
from flask import Flask, Response, jsonify, send_from_directory
from watchdog.events import FileSystemEventHandler, FileClosedEvent
from watchdog.observers import Observer

# ───────────────────────── constants ────────────────────────────────
SR        = 48_000     # Hz
CH        = 2          # stereo
BITS      = 16         # 16‑bit LE PCM
BPS       = SR * CH * (BITS // 8)  # 192 000 bytes/s
CHUNK_SEC = 0.25
CHUNK_FR  = int(SR * CHUNK_SEC)
STABLE_FOR_SEC = 1.0
CROSSFADE_SEC = 1.5
CROSSFADE_FRAMES = int(SR * CROSSFADE_SEC) if CROSSFADE_SEC > 0 else 0
WATCH_DIR_PATH: Path | None = None # Will be set in main

# ───────────────────────── logging / app ────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger("radio")
app = Flask(__name__)

# ───────────────────────── state ────────────────────────────────────
# ### MODIFIED ###: Playlist will store tuples of (Path, int_frames)
playlist: list[tuple[Path, int]] = []
pl_lock  = Lock()
file_ready = Event()
stop_evt   = Event()

sse_subscribers: list[queue.Queue[str]] = []
sse_lock = Lock()

# ───────────────────────── helpers ──────────────────────────────────

def make_wav_header() -> bytes:
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 0xFFFFFFFF, b"WAVE",
        b"fmt ", 16,
        1, CH, SR, BPS,
        CH * (BITS // 8), BITS,
        b"data", 0xFFFFFFFF,
    )

def wait_until_stable(p: Path) -> bool:
    if not p.exists(): return False
    try:
        last_size = p.stat().st_size
        stable_t  = time.time()
        while time.time() - stable_t < STABLE_FOR_SEC:
            time.sleep(0.1)
            if not p.exists(): return False
            cur_size = p.stat().st_size
            if cur_size != last_size:
                last_size = cur_size
                stable_t  = time.time()
        log.debug(f"File {p.name} is stable with size {last_size}.")
        return True
    except FileNotFoundError:
        log.warning(f"File {p.name} disappeared while checking stability.")
        return False
    except Exception as e:
        log.error(f"Error checking stability for {p.name}: {e}")
        return False

def open_when_ready(path: Path, max_wait: float = 15.0):
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            if not wait_until_stable(path):
                time.sleep(0.5)
                continue
            sf_obj = sf.SoundFile(path, "r")
            if sf_obj.samplerate != SR or sf_obj.channels != CH:
                log.error(f"⚠️ {path.name}: wrong SR/CH ({sf_obj.samplerate}/{sf_obj.channels}). Skipped opening.")
                sf_obj.close()
                return None
            return sf_obj
        except (RuntimeError, sf.LibsndfileError) as e:
            log.warning(f"Retrying open for {path.name} due to: {e}")
            time.sleep(0.3)
        except FileNotFoundError:
            log.error(f"⚠️  Skipping {path.name}: File not found during open attempt.")
            return None
    log.error(f"⚠️  Skipping {path.name}: unreadable after {max_wait}s")
    return None

# ───────────────────────── watchdog handler ─────────────────────────
class NewPredFlac(FileSystemEventHandler):
    def on_closed(self, ev: FileClosedEvent):
        if ev.is_directory: return
        p = Path(ev.src_path)
        if not (p.name.startswith("pred_") and p.suffix == ".flac"): return

        log.info(f"Detected potential new file (on_closed): {p.name}")
        if not wait_until_stable(p):
            log.warning(f"File {p.name} was not stable or disappeared. Ignoring.")
            return

        file_frames = 0
        temp_sf = None
        try:
            temp_sf = sf.SoundFile(p, "r")
            if temp_sf.samplerate != SR or temp_sf.channels != CH:
                log.error(f"⚠️ {p.name}: wrong SR/CH ({temp_sf.samplerate}/{temp_sf.channels}). Expected {SR}/{CH}. Skipped.")
                return
            if temp_sf.frames == 0:
                log.warning(f"⚠️ {p.name}: file is empty (0 frames). Skipped.")
                return
            file_frames = temp_sf.frames # ### MODIFIED ###: Get frames here
            log.info(f"File {p.name} format validated: SR={temp_sf.samplerate}, CH={temp_sf.channels}, Frames={file_frames}")
        except (RuntimeError, sf.LibsndfileError, FileNotFoundError) as e:
            log.error(f"⚠️ Error validating {p.name} post-stability: {e}. Skipped.")
            return
        finally:
            if temp_sf: temp_sf.close()

        resolved_path = p.resolve()
        with pl_lock:
            # Check if the file (as a path or as first element of tuple) is already in playlist
            is_already_in_playlist = False
            relative_resolved_path_check = resolved_path # Default to resolved path for check
            if WATCH_DIR_PATH:
                try:
                    relative_resolved_path_check = resolved_path.relative_to(WATCH_DIR_PATH)
                except ValueError:
                    pass # Keep resolved_path if not relative

            for item in playlist:
                item_path = item[0] if isinstance(item, tuple) else item # Handle mixed types during transition if any
                if item_path == relative_resolved_path_check:
                    is_already_in_playlist = True
                    break
                # Fallback for absolute paths possibly stored before logic change
                if item_path == resolved_path:
                    is_already_in_playlist = True
                    break


            if not is_already_in_playlist:
                relative_path_to_store = resolved_path
                if WATCH_DIR_PATH:
                    try:
                        relative_path_to_store = resolved_path.relative_to(WATCH_DIR_PATH)
                    except ValueError:
                        log.warning(f"Could not make {resolved_path} relative to {WATCH_DIR_PATH}. Storing absolute path.")
                
                # ### MODIFIED ###: Append tuple (path, frames)
                playlist.append((relative_path_to_store, file_frames))
                log.info(f"🎵  Queued new latest: {relative_path_to_store.name} ({file_frames} frames)")
                file_ready.set()
                with sse_lock:
                    log.info(f"Notifying {len(sse_subscribers)} SSE subscribers of new song.")
                    for q_item in sse_subscribers: q_item.put("new_song")
            else:
                log.info(f"File {p.name} (resolved: {resolved_path}) already in playlist or a duplicate. Ignoring.")


def start_watcher(folder: Path):
    global WATCH_DIR_PATH
    WATCH_DIR_PATH = folder.resolve()

    obs = Observer()
    event_handler = NewPredFlac()
    obs.schedule(event_handler, str(WATCH_DIR_PATH), recursive=True)
    obs.start()
    log.info(f"👀 Watchdog started on {WATCH_DIR_PATH}")
    try:
        while not stop_evt.is_set(): time.sleep(0.5)
    except KeyboardInterrupt: log.info("Watcher interrupted by user.")
    finally:
        log.info("Stopping watchdog observer..."); obs.stop(); obs.join()
        log.info("Watchdog observer stopped.")

# ───────────────────────── stream generator ─────────────────────────
def radio_loop_latest_song_bytes():
    current_song_tuple: tuple[Path, int] | None = None # ### MODIFIED ###: To store (path, frames)
    sf_file: sf.SoundFile | None = None

    while not stop_evt.is_set():
        with pl_lock:
            if playlist:
                new_latest_song_tuple = playlist[-1]
                if new_latest_song_tuple != current_song_tuple: # Compare tuples
                    current_song_tuple = new_latest_song_tuple
                    if sf_file: sf_file.close()
                    sf_file = None
                    log.info(f"Stream: Attempting to load new song: {current_song_tuple[0].name}")
                break 
        log.info("Stream: Playlist is empty. Waiting for the first track...")
        if file_ready.wait(timeout=1.0): file_ready.clear()
        if stop_evt.is_set(): log.info("Stream: Stop event while waiting for first song."); return

    if not current_song_tuple or stop_evt.is_set():
        log.info("Stream: No song available or stop event. Closing stream."); return

    current_song_rel_path = current_song_tuple[0]
    total_frames = current_song_tuple[1] # ### MODIFIED ###: Get frames from tuple

    current_song_abs_path = WATCH_DIR_PATH / current_song_rel_path if WATCH_DIR_PATH and not current_song_rel_path.is_absolute() else current_song_rel_path

    if not sf_file:
        sf_file = open_when_ready(current_song_abs_path)
        if not sf_file:
            log.error(f"Stream: Failed to open {current_song_abs_path.name}. Client will need to reconnect.");
            with pl_lock: # Remove problematic song from playlist
                # Need to find and remove the tuple carefully
                item_to_remove = None
                for item in playlist:
                    if item[0] == current_song_rel_path:
                        item_to_remove = item
                        break
                if item_to_remove and item_to_remove in playlist: # Check if it's still the same tuple
                     playlist.remove(item_to_remove)
                     log.info(f"Stream: Removed ({current_song_rel_path.name}, {total_frames} frames) from playlist due to open failure.")
                current_song_tuple = None # Ensure we don't try to use it again in this run
            return 
    
    # total_frames was already set from current_song_tuple[1]
    # We can verify if sf_file.frames matches, but primarily use the one from playlist for consistency
    if sf_file.frames != total_frames:
        log.warning(f"Stream: Mismatch in frame count for {current_song_abs_path.name}. Playlist: {total_frames}, File: {sf_file.frames}. Using playlist value.")
        # total_frames remains as from playlist

    log.info(f"Stream: Starting to play and loop {current_song_abs_path.name} ({total_frames} frames)")

    can_crossfade = CROSSFADE_FRAMES > 0 and total_frames >= CROSSFADE_FRAMES * 1.1
    if not can_crossfade:
        log.info(f"Stream: Crossfading disabled for {current_song_abs_path.name} (too short or CF disabled). Frames: {total_frames}, CF_Frames: {CROSSFADE_FRAMES}")

    yield make_wav_header()

    try:
        current_frame_in_song = 0
        sf_file.seek(0)

        while not stop_evt.is_set():
            loop_start_time = time.perf_counter()

            with pl_lock:
                if playlist and playlist[-1][0] != current_song_rel_path: # Compare paths in tuple
                    log.info(f"Stream: New song {playlist[-1][0].name} detected. Current song {current_song_rel_path.name} will stop for this client.")
                    return

            frames_left_in_song = total_frames - current_frame_in_song

            if can_crossfade and frames_left_in_song <= CROSSFADE_FRAMES:
                log.debug(f"Stream: Entering crossfade for {current_song_abs_path.name}. Pos: {current_frame_in_song}, Left: {frames_left_in_song}")
                sf_file.seek(current_frame_in_song)
                tail_read_len = min(frames_left_in_song, CROSSFADE_FRAMES) 
                tail_data_float = sf_file.read(tail_read_len, dtype="float32", always_2d=True)

                sf_file.seek(0)
                head_read_len = min(total_frames, CROSSFADE_FRAMES) 
                head_data_float = sf_file.read(head_read_len, dtype="float32", always_2d=True)

                actual_cf_len = min(len(tail_data_float), len(head_data_float), CROSSFADE_FRAMES)
                if actual_cf_len == 0:
                    log.warning(f"Stream: Zero length data for crossfade in {current_song_abs_path.name}. Skipping crossfade.");
                    current_frame_in_song = 0
                    sf_file.seek(0)
                    continue

                tail_data_float = tail_data_float[:actual_cf_len]
                head_data_float = head_data_float[:actual_cf_len]

                p_gain = np.linspace(0.0, 1.0, actual_cf_len, endpoint=True) 
                gain_in_ramp = np.sin(p_gain * np.pi / 2.0)[:, np.newaxis]
                gain_out_ramp = np.cos(p_gain * np.pi / 2.0)[:, np.newaxis]

                mixed_frames_offset = 0
                while mixed_frames_offset < actual_cf_len and not stop_evt.is_set():
                    chunk_s_time = time.perf_counter()
                    frames_to_mix = min(CHUNK_FR, actual_cf_len - mixed_frames_offset)

                    sub_tail = tail_data_float[mixed_frames_offset : mixed_frames_offset + frames_to_mix]
                    sub_head = head_data_float[mixed_frames_offset : mixed_frames_offset + frames_to_mix]
                    sub_gain_out = gain_out_ramp[mixed_frames_offset : mixed_frames_offset + frames_to_mix]
                    sub_gain_in  = gain_in_ramp[mixed_frames_offset : mixed_frames_offset + frames_to_mix]

                    mixed_chunk_float = (sub_tail * sub_gain_out) + (sub_head * sub_gain_in)
                    np.clip(mixed_chunk_float, -1.0, 1.0, out=mixed_chunk_float)
                    mixed_chunk_int16 = (mixed_chunk_float * 32767).astype(np.int16)

                    yield mixed_chunk_int16.tobytes()
                    mixed_frames_offset += frames_to_mix

                    actual_chunk_duration_sec = frames_to_mix / SR
                    elapsed_for_chunk_sec = time.perf_counter() - chunk_s_time
                    sleep_duration = max(0, actual_chunk_duration_sec - elapsed_for_chunk_sec)
                    if sleep_duration > 0.001: time.sleep(sleep_duration)

                current_frame_in_song = mixed_frames_offset 
                sf_file.seek(current_frame_in_song)
                log.debug(f"Stream: Crossfade finished. Resuming {current_song_abs_path.name} from frame {current_frame_in_song}")
            else:
                frames_to_read_now = min(CHUNK_FR, total_frames - current_frame_in_song)

                if frames_to_read_now <= 0: 
                    if not can_crossfade: 
                        log.debug(f"Stream: Simple loop for {current_song_abs_path.name}. Seeking to 0.")
                        sf_file.seek(0)
                        current_frame_in_song = 0
                        frames_to_read_now = min(CHUNK_FR, total_frames) 
                    else: 
                        log.warning(f"Stream: Reached EOF for {current_song_abs_path.name} when crossfade was expected but not triggered. Breaking loop for safety.")
                        break 

                sf_file.seek(current_frame_in_song) 
                frames_data = sf_file.read(frames_to_read_now, dtype="int16", always_2d=True)

                if len(frames_data) == 0:
                    if not stop_evt.is_set(): log.warning(f"Stream: Read 0 frames unexpectedly from {current_song_abs_path.name}. Breaking.");
                    break

                yield frames_data.tobytes()
                current_frame_in_song += len(frames_data)

                if current_frame_in_song >= total_frames and not can_crossfade: 
                     current_frame_in_song = 0
                     
                actual_chunk_duration_sec = len(frames_data) / SR
                elapsed_for_chunk_sec = time.perf_counter() - loop_start_time
                sleep_duration = max(0, actual_chunk_duration_sec - elapsed_for_chunk_sec)
                if sleep_duration > 0.001 : time.sleep(sleep_duration)

    except GeneratorExit:
        log.info(f"Stream: Client disconnected from {current_song_abs_path.name if current_song_abs_path and hasattr(current_song_abs_path, 'name') else 'N/A'}")
    except sf.LibsndfileError as e:
        log.error(f"Stream: LibsndfileError for {current_song_abs_path.name if current_song_abs_path and hasattr(current_song_abs_path, 'name') else 'N/A'}: {e}.")
    except RuntimeError as e:
        log.error(f"Stream: RuntimeError for {current_song_abs_path.name if current_song_abs_path and hasattr(current_song_abs_path, 'name') else 'N/A'}: {e}.")
    except Exception as e:
        log.error(f"Stream: Unexpected error for {current_song_abs_path.name if current_song_abs_path and hasattr(current_song_abs_path, 'name') else 'N/A'}: {type(e).__name__} {e}.")
    finally:
        if sf_file:
            log.info(f"Stream: Closing soundfile for {current_song_abs_path.name if current_song_abs_path and hasattr(current_song_abs_path, 'name') else 'N/A'}")
            sf_file.close()

# ─────────────────── Server-Sent Events (SSE) ───────────────────────
def sse_event_stream():
    q = queue.Queue()
    with sse_lock:
        sse_subscribers.append(q)
        log.info(f"SSE: New client connected. Total subscribers: {len(sse_subscribers)}")
    try:
        while not stop_evt.is_set():
            try:
                message = q.get(timeout=1.0)
                if message == "__close__": break
                yield f"data: {message}\n\n"
            except queue.Empty:
                if stop_evt.is_set(): break
                continue
    except GeneratorExit:
        log.info("SSE: Client disconnected (GeneratorExit).")
    except Exception as e:
        log.error(f"SSE: Error in event stream: {e}")
    finally:
        with sse_lock:
            if q in sse_subscribers: sse_subscribers.remove(q)
            log.info(f"SSE: Client queue removed. Total subscribers: {len(sse_subscribers)}")

@app.route('/events')
def events():
    return Response(sse_event_stream(), mimetype='text/event-stream',
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ───────────────────────── Flask routes ─────────────────────────────
@app.route("/stream")
def stream_route():
    return Response(radio_loop_latest_song_bytes(), mimetype="audio/wav",
                    headers={"Cache-Control":"no-cache"})

@app.route("/get_audio_file/<path:filename>")
def get_audio_file_route(filename):
    if not WATCH_DIR_PATH:
        log.error("/get_audio_file: WATCH_DIR_PATH not set.")
        return "Server configuration error", 500
    
    # filename is expected to be a relative path string from WATCH_DIR_PATH
    log.info(f"Serving full audio file: {filename} from {WATCH_DIR_PATH}")
    try:
        # Ensure the filename is treated as relative to WATCH_DIR_PATH
        # Path() constructor with a single argument might treat it as absolute if it starts with /
        # Instead, directly join WATCH_DIR_PATH with filename
        requested_path = (WATCH_DIR_PATH / filename).resolve() # Resolve to get canonical path

        # Security check: ensure resolved path is within WATCH_DIR_PATH
        if WATCH_DIR_PATH.resolve() not in requested_path.parents and requested_path != WATCH_DIR_PATH.resolve():
             log.warning(f"Attempt to access file outside watch directory: {filename} (resolved: {requested_path}) vs {WATCH_DIR_PATH.resolve()}")
             return "Access denied", 403
        
        # send_from_directory expects the directory and then the filename relative to that directory
        return send_from_directory(str(WATCH_DIR_PATH.resolve()), filename, as_attachment=False)
    except FileNotFoundError:
        log.error(f"File not found for /get_audio_file: {filename}")
        return "File not found", 404
    except Exception as e:
        log.error(f"Error serving file {filename}: {e}")
        return "Server error", 500


@app.route("/latestfile")
def latest_file_api():
    with pl_lock:
        if playlist:
            latest_item = playlist[-1]  # Now a tuple (path, frames)
            latest_relative_path = latest_item[0]
            song_total_frames_val = latest_item[1]
            
            return jsonify({
                "latest_file": latest_relative_path.name,
                "latest_file_path": str(latest_relative_path), # Path is relative
                "song_total_frames": song_total_frames_val,
                "crossfade_frames": CROSSFADE_FRAMES, 
                "sample_rate": SR 
            })
        return jsonify({
            "latest_file": None,
            "latest_file_path": None,
            "song_total_frames": 0,
            "crossfade_frames": 0,
            "sample_rate": SR
        }), 404

@app.route("/")
def index():
  return """<!doctype html>
<html>
<head>
    <title>LoRAdio</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        /* Basic reset and full page setup */
        html, body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol";
            color: #e0e0e0;
        }

        #full-spectrogram-bg-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: -1;
        }
        #fullSpectrogramCanvas, #playheadCanvas {
            display: block;
            width: 100%;
            height: 100%;
        }
        #fullSpectrogramCanvas {
            background-color: #080808;
        }
        #playheadCanvas {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
        }

        #app-content-overlay {
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
            width: 100%;
            height: 100vh;
            padding: 20px;
            box-sizing: border-box;
            overflow-y: auto;
            background-color: rgba(18, 18, 18, 0.25);
            -webkit-backdrop-filter: blur(0px);
            backdrop-filter: blur(0px);
            transition: background-color 0.3s ease, backdrop-filter 0.3s ease; /* Added transition */
        }

        h1 {
            color: #1DB954;
            margin-bottom: 20px;
            font-size: 2em;
            text-shadow: 0 0 8px rgba(29, 185, 84, 0.7);
            text-align: center;
            transition: font-size 0.3s ease, margin 0.3s ease, padding 0.3s ease; /* Added transition */
        }

        #controls {
            text-align: center;
            margin-bottom: 20px;
            flex-shrink: 0;
        }
        button {
            margin: 0 10px;
            padding: 12px 25px;
            font-size: 1.1rem;
            border: none;
            border-radius: 25px;
            background-color: #1DB954;
            color: white;
            font-weight: bold;
            cursor: pointer;
            transition: background-color 0.2s ease, transform 0.1s ease;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        button:hover { background-color: #1ed760; }
        button:active { transform: scale(0.97); }
        button:disabled {
            background-color: #535353;
            color: #ababab;
            cursor: not-allowed;
            opacity: 0.8;
        }

        #filter-controls {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 15px;
            width: 100%;
            max-width: 580px;
            margin-bottom: 25px;
            padding: 20px;
            background-color: rgba(40, 40, 40, 0.9);
            border-radius: 10px;
            box-sizing: border-box;
            border: 1px solid #4a4a4a;
            flex-shrink: 0;
            box-shadow: 0 3px 8px rgba(0,0,0,0.3);
        }
        .filter-row {
            display: flex;
            justify-content: space-around;
            width: 100%;
            gap: 15px;
            flex-wrap: wrap;
        }
        .filter-group {
            display: flex;
            flex-direction: column;
            align-items: center;
            font-size: 0.95em;
            flex: 1;
            min-width: 160px;
        }
        .filter-group label {
            margin-bottom: 8px;
            color: #b3b3b3;
            font-weight: 500;
            text-align: center;
        }
        .filter-group input[type="range"] {
            width: 100%;
            max-width: 200px;
            cursor: pointer;
            margin-top: 5px;
            accent-color: #1DB954;
        }
        #highpass-value, #lowpass-value, #yMinFreqValue, #yMaxFreqValue {
            font-weight: bold;
            color: #1DB954;
            min-width: 55px;
            display: inline-block;
            text-align: right;
            background-color: #333;
            padding: 3px 6px;
            border-radius: 4px;
            font-size: 0.9em;
        }
         @media (max-width: 700px) {
            .filter-row {
                flex-direction: column;
                align-items: center;
                gap: 18px;
            }
            .filter-group { width: 90%; max-width: 300px; }
            h1 { font-size: 1.8em; }
            button { padding: 10px 20px; font-size: 1rem; }
            #filter-controls { padding: 15px; }
        }

        #current_song_display {
            margin-top: 0;
            margin-bottom: 18px;
            font-style: italic;
            color: #b3b3b3;
            min-height: 1.2em;
            text-align: center;
            font-size: 1em;
            flex-shrink: 0;
            background-color: rgba(0,0,0,0.3);
            padding: 5px 10px;
            border-radius: 5px;
            transition: all 0.3s ease; /* Added transition */
        }
        #media-player-section {
            width: 100%;
            max-width: 450px;
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-bottom: 20px;
            flex-shrink: 0;
        }
        audio {
            width: 100%;
            margin-bottom: 15px;
            border-radius: 5px;
        }
           #spectrogramCanvas {
            display: block; /* Ensure it's a block for margin auto to work */
            width: 100%; /* Take available width */
            max-width: 450px; /* But cap it, similar to old media-player-section */
            height: 100px;
            background-color: #000; /* Default background when settings are visible */
            border-radius: 4px;
            box-shadow: inset 0 0 5px rgba(0,0,0,0.5);
            margin: 0 auto 20px auto; /* Center it and provide bottom margin */
            /* flex-shrink: 0; /* uncomment if needed, but margin auto should work with parent's align-items:center */
        }
        #spectrogram-status {
            position: fixed;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.85em;
            color: #aaa;
            background-color: rgba(0,0,0,0.6);
            padding: 5px 12px;
            border-radius: 15px;
            min-height: 1em;
            text-align: center;
            z-index: 10; /* Ensure it's above normal content */
        }
  /* Styling for #spectrogramCanvas when settings are hidden */
        #app-content-overlay.settings-hidden > #spectrogramCanvas {
            position: fixed;
            top: 115px; /* MOVED TO TOP: Below #current_song_display */
                       /* (70px + ~35-40px for song display + small gap) */
            left: 50%;
            transform: translateX(-50%);
            width: 90%; /* Responsive width */
            max-width: 400px; /* Max width in hidden mode */
            height: 80px; /* Slightly smaller height in hidden mode */
            z-index: 15;  /* Same level as h1 and song_display */
            background-color: rgba(0, 0, 0, 0.7); /* Background for hidden mode */
            border-radius: 4px;
            box-shadow: 0 0 10px rgba(0,0,0,0.5);
            margin: 0; /* Reset margins */
        }
        /* --- Styles for Hide/Show Settings --- */
        #toggleSettingsBtn {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 20; /* High z-index to be on top */
            padding: 8px 15px;
            font-size: 0.9rem;
            background-color: #1DB954;
            color: white;
            border: none;
            border-radius: 20px;
            cursor: pointer;
            box-shadow: 0 2px 5px rgba(0,0,0,0.3);
            margin: 0; /* Reset margin */
        }
        #toggleSettingsBtn:hover {
            background-color: #1ed760;
        }

        #app-content-overlay.settings-hidden {
            background-color: transparent;
            -webkit-backdrop-filter: none;
            backdrop-filter: none;
            padding: 0;
            overflow-y: hidden; /* Prevent scrolling when hidden */
        }

        #app-content-overlay.settings-hidden > #controls,
        #app-content-overlay.settings-hidden > #filter-controls,
        #app-content-overlay.settings-hidden > #media-player-section {
            display: none;
        }

        #app-content-overlay.settings-hidden > h1 {
            position: fixed;
            top: 20px;
            left: 20px;
            font-size: 1.2em; /* Smaller font size */
            color: #1DB954;
            text-shadow: 0 0 5px rgba(29, 185, 84, 0.5);
            background-color: rgba(18, 18, 18, 0.65);
            padding: 6px 12px;
            border-radius: 8px;
            margin: 0; /* Reset margin */
            z-index: 15; /* Above overlay, below toggle button */
        }

          #app-content-overlay.settings-hidden > #current_song_display {
            position: fixed;
            top: 70px; /* MOVED TO TOP: Approx. below h1/toggle button */
            left: 50%;
            transform: translateX(-50%);
            font-size: 1em;
            color: #e0e0e0;
            background-color: rgba(0, 0, 0, 0.75);
            padding: 8px 15px; /* Slightly more compact padding */
            border-radius: 8px;
            z-index: 15;
            text-align: center;
            width: auto; /* Adjust width based on content */
            max-width: 70%; /* Max width for top display */
            margin: 0; /* Reset margins */
            min-height: auto; /* Reset min-height */
        }
    </style>
</head>
<body>

<div id="full-spectrogram-bg-container">
    <canvas id="fullSpectrogramCanvas"></canvas>
    <canvas id="playheadCanvas"></canvas>
</div>

<div id="app-content-overlay">
    <h1>LoRAdio</h1>
    <div id="controls">
      <button id="play">Listen</button>
      <button id="stop" disabled>Stop Listening</button>
    </div>

    <div id="filter-controls">
        <div class="filter-row">
            <div class="filter-group">
                <label for="highpass-slider">Audio HP: <span id="highpass-value">20</span> Hz</label>
                <input type="range" id="highpass-slider" min="20" max="10000" step="10" value="20">
            </div>
            <div class="filter-group">
                <label for="lowpass-slider">Audio LP: <span id="lowpass-value">20000</span> Hz</label>
                <input type="range" id="lowpass-slider" min="100" max="20000" step="100" value="20000">
            </div>
        </div>
        <div class="filter-row">
             <div class="filter-group">
                <label for="yMinFreqSlider">Spec. Min Freq: <span id="yMinFreqValue">0</span> Hz</label>
                <input type="range" id="yMinFreqSlider" min="0" max="99" step="1" value="0">
            </div>
            <div class="filter-group">
                <label for="yMaxFreqSlider">Spec. Max Freq: <span id="yMaxFreqValue">24000</span> Hz</label>
                <input type="range" id="yMaxFreqSlider" min="1" max="100" step="1" value="100">
            </div>
        </div>
    </div>

<p id="current_song_display">No song loaded.</p>

    <div id="media-player-section"> <audio id="aud" controls></audio>
    </div>
    <canvas id="spectrogramCanvas"></canvas>

    <p id="spectrogram-status"></p>
</div>

<script>

// DOM Elements
const audio = document.getElementById('aud');
const playBtn = document.getElementById('play');
const stopBtn = document.getElementById('stop');
const songDisplay = document.getElementById('current_song_display');

// Live Spectrogram
const spectrogramCanvas = document.getElementById('spectrogramCanvas');
const canvasCtx = spectrogramCanvas.getContext('2d');

// Filter Controls
const highpassSlider = document.getElementById('highpass-slider');
const highpassValueDisplay = document.getElementById('highpass-value');
const lowpassSlider = document.getElementById('lowpass-slider');
const lowpassValueDisplay = document.getElementById('lowpass-value');

// Spectrogram Y-Axis Window Controls
const yMinFreqSlider = document.getElementById('yMinFreqSlider');
const yMinFreqValueDisplay = document.getElementById('yMinFreqValue');
const yMaxFreqSlider = document.getElementById('yMaxFreqSlider');
const yMaxFreqValueDisplay = document.getElementById('yMaxFreqValue');

// Full Screen Background Spectrogram & Playhead
const fullSpectrogramCanvas = document.getElementById('fullSpectrogramCanvas');
const fullSpectrogramCtx = fullSpectrogramCanvas.getContext('2d');
const playheadCanvas = document.getElementById('playheadCanvas');
const playheadCtx = playheadCanvas.getContext('2d');
const spectrogramStatusDisplay = document.getElementById('spectrogram-status');

// ### NEW ###: App Overlay and Title for Toggling
const appContentOverlay = document.getElementById('app-content-overlay');
const h1Title = document.querySelector('#app-content-overlay > h1');
let toggleSettingsBtn; // Will be created in initializeApp


// Audio Constants & State
const FADE_DURATION_MS = 1000;
const FADE_INTERVAL_MS = 50;
let fadeIntervalId = null;

let audioCtx = null;
let analyser = null;
let sourceNode = null;
let dataArray = null;
let liveAnimationId = null;
let highpassFilterNode = null;
let lowpassFilterNode = null;

let serverEffectiveSongDurationSec = 0;
let serverEffectiveCrossfadeDurationSec = 0;

let originalFullSongAudioBuffer = null;
let fullSongSpectrogramDataMatrix = null;
let currentSongFilePath = null;
let currentSongDuration = 0; 

const DEFAULT_HIGHPASS_FREQ = 20;
const DEFAULT_LOWPASS_FREQ = 20000;
const DEFAULT_SR = 48000; 

let yMinFreqPercent = 0.0;
let yMaxFreqPercent = 1.0;

const FULL_SPECTROGRAM_FFT_SIZE = 8192;
const FULL_SPECTROGRAM_TARGET_SLICES = 1200;

const VIRIDIS_COLORS = [
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
    [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37]
].map(c => `rgb(${c[0]},${c[1]},${c[2]})`);

// --- Utility Functions ---
function clearExistingFade() {
    if (fadeIntervalId !== null) {
        clearInterval(fadeIntervalId);
        fadeIntervalId = null;
    }
}

function fetchAndSetLatestSongName() {
    return fetch('/latestfile')
        .then(response => response.json())
        .then(data => {
            if (data.latest_file && data.sample_rate > 0 && data.song_total_frames !== undefined && data.crossfade_frames !== undefined) {
                songDisplay.textContent = "Now Playing: " + data.latest_file;
                currentSongFilePath = data.latest_file_path; 

                const sRate = data.sample_rate;
                const C_frames = data.crossfade_frames;
                const D_frames = data.song_total_frames;

                serverEffectiveCrossfadeDurationSec = C_frames / sRate;
                serverEffectiveSongDurationSec = D_frames / sRate;
                
                console.log(`Server effective D: ${serverEffectiveSongDurationSec.toFixed(4)}s, C: ${serverEffectiveCrossfadeDurationSec.toFixed(4)}s (from frames)`);
                
            } else {
                songDisplay.textContent = "Waiting for a song...";
                currentSongFilePath = null;
                serverEffectiveSongDurationSec = 0;
                serverEffectiveCrossfadeDurationSec = 0;
                 if(data.sample_rate <=0) console.warn("Received invalid sample_rate from server:", data.sample_rate);
            }
        })
        .catch(error => {
            console.error("Error fetching latest song name:", error);
            songDisplay.textContent = "Could not fetch song info.";
            currentSongFilePath = null;
            serverEffectiveSongDurationSec = 0;
            serverEffectiveCrossfadeDurationSec = 0;
        });
}

function fadeAudio(targetVolume, duration, callback) {
    clearExistingFade();
    const startVolume = audio.volume;
    const diff = targetVolume - startVolume;
    if (diff === 0 && targetVolume === audio.volume) {
        if (callback) callback(); return;
    }
    const steps = Math.max(1, duration / FADE_INTERVAL_MS);
    const increment = diff / steps;
    let currentStep = 0;

    fadeIntervalId = setInterval(() => {
        currentStep++;
        let newVolume = startVolume + (increment * currentStep);
        newVolume = Math.max(0, Math.min(1, newVolume));

        if ((increment > 0 && newVolume >= targetVolume) || (increment < 0 && newVolume <= targetVolume) || currentStep >= steps) {
            audio.volume = targetVolume;
            clearExistingFade();
            if (targetVolume === 0 && audio.src && !audio.paused) audio.pause();
            if (callback) callback();
        } else {
            audio.volume = newVolume;
        }
    }, FADE_INTERVAL_MS);
}

// --- Canvas Sizing and Drawing ---
function resizeCanvases() {
    const isSettingsHidden = appContentOverlay.classList.contains('settings-hidden');

    // Live Spectrogram Canvas (#spectrogramCanvas)
    if (isSettingsHidden) {
        // Settings are hidden: #spectrogramCanvas is fixed
        const targetWidth = Math.min(window.innerWidth * 0.9, 400); // CSS: width: 90%, max-width: 400px
        spectrogramCanvas.width = Math.max(10, targetWidth);
        spectrogramCanvas.height = 80; // CSS: height: 80px
    } else {
        // Settings are visible: #spectrogramCanvas in normal flow
        // CSS: width: 100%, max-width: 450px, centered in appContentOverlay.
        // appContentOverlay is flex, align-items:center. Canvas takes actual width up to max-width.
        const canvasEffectiveWidth = Math.min(appContentOverlay.clientWidth, 450);
        spectrogramCanvas.width = Math.max(10, canvasEffectiveWidth);
        spectrogramCanvas.height = 100; // CSS: height: 100px
    }

    // Fallback fill for #spectrogramCanvas if live animation isn't running
    // This primarily affects the initial state or after stopping.
    if (!liveAnimationId && canvasCtx && spectrogramCanvas.width > 0) {
         if (isSettingsHidden) {
            // In hidden mode, the CSS provides a semi-transparent background.
            // Clearing to transparent ensures that background is visible.
            canvasCtx.clearRect(0, 0, spectrogramCanvas.width, spectrogramCanvas.height);
         } else {
            // In normal mode, it has a black background.
            canvasCtx.fillStyle = '#000';
            canvasCtx.fillRect(0, 0, spectrogramCanvas.width, spectrogramCanvas.height);
         }
    }

    // Full screen background spectrogram and playhead (these are always viewport-sized)
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    fullSpectrogramCanvas.width = vw; fullSpectrogramCanvas.height = vh;
    playheadCanvas.width = vw; playheadCanvas.height = vh;

    if (fullSongSpectrogramDataMatrix) {
        drawFullSpectrogram(fullSongSpectrogramDataMatrix);
    } else {
        clearFullSpectrogramDisplay();
    }
    clearPlayhead(); // Clear and redraw playhead after potential resize
}

function mapNormalizedToViridis(normalizedValue) {
    const v = Math.max(0, Math.min(1, normalizedValue));
    if (v === 0) return VIRIDIS_COLORS[0];
    if (v === 1) return VIRIDIS_COLORS[VIRIDIS_COLORS.length - 1];
    const index = Math.min(VIRIDIS_COLORS.length - 1, Math.floor(v * (VIRIDIS_COLORS.length -1)));
    return VIRIDIS_COLORS[index];
}
function drawLiveSpectrogramAndPlayhead() {
    // Live spectrogram part (always draw if analyser is ready and canvas has dimensions)
    if (analyser && canvasCtx && spectrogramCanvas.width > 0 && spectrogramCanvas.height > 0) {
        analyser.getByteFrequencyData(dataArray); // Get current frequency data

        // Handle canvas background:
        // If settings are hidden, #spectrogramCanvas has a semi-transparent CSS background.
        // We need to clear the canvas to transparency to let that CSS background show.
        // If settings are visible, #spectrogramCanvas has a solid black background by default.
        if (appContentOverlay.classList.contains('settings-hidden')) {
            canvasCtx.clearRect(0, 0, spectrogramCanvas.width, spectrogramCanvas.height);
        } else {
            canvasCtx.fillStyle = '#000'; // Default solid black background for normal view
            canvasCtx.fillRect(0, 0, spectrogramCanvas.width, spectrogramCanvas.height);
        }

        // Draw the frequency bars
        const barWidth = (spectrogramCanvas.width / analyser.frequencyBinCount);
        let x = 0;
        for (let i = 0; i < analyser.frequencyBinCount; i++) {
            const normalizedLiveValue = dataArray[i] / 255.0;
            const liveColor = mapNormalizedToViridis(normalizedLiveValue); // Your color mapping function
            const barHeight = normalizedLiveValue * spectrogramCanvas.height;

            canvasCtx.fillStyle = liveColor;
            canvasCtx.fillRect(x, spectrogramCanvas.height - barHeight, barWidth, barHeight);
            x += barWidth;
        }
    }

    // Playhead part (for the full-screen background spectrogram)
    if (playheadCanvas.width > 0 && fullSongSpectrogramDataMatrix && audio.src && !audio.paused && audio.readyState >= 2) {
        clearPlayhead(); // Clear previous playhead drawing

        const tAbs = audio.currentTime; // Absolute current time of the audio element
        const D = serverEffectiveSongDurationSec; // Server-authoritative total song duration
        const C = serverEffectiveCrossfadeDurationSec; // Server-authoritative crossfade duration
        let conceptualSongTime; // The time to map onto the visual spectrogram

        // Calculate conceptual song time considering server-side looping and crossfading logic
        if (D <= 0) { // No duration, or invalid
            conceptualSongTime = 0;
        } else if (C <= 0 || C >= D) { // No crossfade or crossfade is as long/longer than song (simple loop)
            conceptualSongTime = tAbs % D;
        } else { // Crossfading is active and meaningful
            if (tAbs < D) { // Still in the first play-through of the song
                conceptualSongTime = tAbs;
            } else { // Into subsequent loops after the first full play-through
                const timeIntoSubsequentPattern = tAbs - D; // Time elapsed since the first D ended
                const subsequentLoopLength = D - C; // Effective length of loops after the first one

                if (subsequentLoopLength <= 0) { // Should not happen if C < D
                    conceptualSongTime = C; // Fallback, effectively stuck in crossfade region
                } else {
                    const timeWithinCurrentSubsequentLoop = timeIntoSubsequentPattern % subsequentLoopLength;
                    conceptualSongTime = C + timeWithinCurrentSubsequentLoop; // Maps to the part of song after crossfade start
                }
            }
        }

        // Ensure playheadPositionForProgress is always within the 0 to D range for progress calculation
        const playheadPositionForProgress = (D > 0) ? (conceptualSongTime % D) : 0;
        const progress = (D > 0) ? (playheadPositionForProgress / D) : 0; // Normalized progress (0.0 to 1.0)

        const playheadX = progress * playheadCanvas.width; // X-coordinate on the playhead canvas
        const playheadVisualWidth = 8; // Visual width of the playhead line/gradient
        const halfPlayheadVisualWidth = playheadVisualWidth / 2;

        // Create a gradient for a softer playhead appearance
        const gradientStartX = playheadX - halfPlayheadVisualWidth;
        const gradientEndX = playheadX + halfPlayheadVisualWidth;
        const gradient = playheadCtx.createLinearGradient(gradientStartX, 0, gradientEndX, 0);
        const peakR = 255, peakG = 255, peakB = 255; // Playhead color (white)
        const peakAlpha = 0.5; // Max opacity for the playhead center

        gradient.addColorStop(0, `rgba(${peakR}, ${peakG}, ${peakB}, 0)`); // Fade out at edge
        gradient.addColorStop(0.3, `rgba(${peakR}, ${peakG}, ${peakB}, ${peakAlpha * 0.5})`);
        gradient.addColorStop(0.5, `rgba(${peakR}, ${peakG}, ${peakB}, ${peakAlpha})`); // Peak opacity at center
        gradient.addColorStop(0.7, `rgba(${peakR}, ${peakG}, ${peakB}, ${peakAlpha * 0.5})`);
        gradient.addColorStop(1, `rgba(${peakR}, ${peakG}, ${peakB}, 0)`); // Fade out at other edge

        playheadCtx.fillStyle = gradient;
        playheadCtx.fillRect(gradientStartX, 0, playheadVisualWidth, playheadCanvas.height);

    } else if (playheadCanvas.width > 0) { // If playhead should be drawn but conditions not met (e.g., audio stopped)
        clearPlayhead(); // Ensure playhead is cleared
    }

    // Request the next frame for continuous animation
    liveAnimationId = requestAnimationFrame(drawLiveSpectrogramAndPlayhead);
}
function clearFullSpectrogramDisplay() {
    if (fullSpectrogramCtx && fullSpectrogramCanvas.width > 0) {
        fullSpectrogramCtx.fillStyle = '#080808';
        fullSpectrogramCtx.fillRect(0, 0, fullSpectrogramCanvas.width, fullSpectrogramCanvas.height);
    }
}
function clearPlayhead() {
    if (playheadCtx && playheadCanvas.width > 0) {
        playheadCtx.clearRect(0, 0, playheadCanvas.width, playheadCanvas.height);
    }
}

// --- Full Song Spectrogram Generation (REAL STFT) ---
async function fetchFullAudioAndGenerateSpectrogram(filePath) {
    if (!filePath) {
        spectrogramStatusDisplay.textContent = "No file path for spectrogram.";
        return;
    }
    spectrogramStatusDisplay.textContent = "Loading full song data...";
    originalFullSongAudioBuffer = null;
    fullSongSpectrogramDataMatrix = null;
    currentSongDuration = 0; 
    clearFullSpectrogramDisplay();
    clearPlayhead();

    try {
        const response = await fetch(`/get_audio_file/${encodeURIComponent(filePath)}`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status} for ${filePath}`);
        const arrayBuffer = await response.arrayBuffer();
        spectrogramStatusDisplay.textContent = "Decoding audio data...";

        if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        originalFullSongAudioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        currentSongDuration = originalFullSongAudioBuffer.duration; 
        console.log(`Decoded audio duration (currentSongDuration for spectrogram): ${currentSongDuration.toFixed(4)}s`);
        updateSpectrogramFrequencyWindowControls();

        spectrogramStatusDisplay.textContent = "Filtering audio for spectrogram...";
        const offlineCtx = new OfflineAudioContext(
            originalFullSongAudioBuffer.numberOfChannels,
            originalFullSongAudioBuffer.length,
            originalFullSongAudioBuffer.sampleRate
        );
        const offlineSource = offlineCtx.createBufferSource();
        offlineSource.buffer = originalFullSongAudioBuffer;
        const offlineHighpass = offlineCtx.createBiquadFilter();
        offlineHighpass.type = "highpass";
        offlineHighpass.frequency.value = parseFloat(highpassSlider.value);
        offlineHighpass.Q.value = 1;
        const offlineLowpass = offlineCtx.createBiquadFilter();
        offlineLowpass.type = "lowpass";
        offlineLowpass.frequency.value = parseFloat(lowpassSlider.value);
        offlineLowpass.Q.value = 1;
        offlineSource.connect(offlineHighpass);
        offlineHighpass.connect(offlineLowpass);
        offlineLowpass.connect(offlineCtx.destination);
        offlineSource.start();
        const filteredBuffer = await offlineCtx.startRendering();
        spectrogramStatusDisplay.textContent = "Generating spectrogram data (STFT)...";

        const channelData = filteredBuffer.getChannelData(0);
        const songLengthFrames = filteredBuffer.length;
        const numRawFrequencyBins = FULL_SPECTROGRAM_FFT_SIZE / 2;

        let hopSize;
        const analyzableFrames = songLengthFrames - FULL_SPECTROGRAM_FFT_SIZE;
        if (FULL_SPECTROGRAM_TARGET_SLICES <= 1) hopSize = 0;
        else hopSize = (analyzableFrames <=0) ? 1 : Math.max(1, Math.floor(analyzableFrames / (FULL_SPECTROGRAM_TARGET_SLICES - 1)));

        if (songLengthFrames < FULL_SPECTROGRAM_FFT_SIZE) {
             console.warn(`Song length (${songLengthFrames}) shorter than FFT_SIZE (${FULL_SPECTROGRAM_FFT_SIZE}).`);
        }

        fullSongSpectrogramDataMatrix = Array(FULL_SPECTROGRAM_TARGET_SLICES).fill(null).map(() => new Float32Array(numRawFrequencyBins));
        const tempFreqDataArray = new Uint8Array(numRawFrequencyBins);
        const fftSegmentBuffer = audioCtx.createBuffer(1, FULL_SPECTROGRAM_FFT_SIZE, filteredBuffer.sampleRate);
        const fftSegmentChannelData = fftSegmentBuffer.getChannelData(0);

        for (let t = 0; t < FULL_SPECTROGRAM_TARGET_SLICES; t++) {
            if (t % Math.floor(FULL_SPECTROGRAM_TARGET_SLICES / 20) === 0) {
                spectrogramStatusDisplay.textContent = `Generating spectrogram: ${Math.round((t / FULL_SPECTROGRAM_TARGET_SLICES) * 100)}%`;
                await new Promise(resolve => setTimeout(resolve, 0));
            }
            const startFrameInSong = t * hopSize;
            fftSegmentChannelData.fill(0.0);
            const segmentFromSong = channelData.subarray(startFrameInSong, startFrameInSong + FULL_SPECTROGRAM_FFT_SIZE);
            fftSegmentChannelData.set(segmentFromSong);

            const sliceOfflineCtx = new OfflineAudioContext(1, FULL_SPECTROGRAM_FFT_SIZE, filteredBuffer.sampleRate);
            const sliceSource = sliceOfflineCtx.createBufferSource();
            sliceSource.buffer = fftSegmentBuffer;
            const sliceAnalyser = sliceOfflineCtx.createAnalyser();
            sliceAnalyser.fftSize = FULL_SPECTROGRAM_FFT_SIZE;
            sliceAnalyser.smoothingTimeConstant = 0;
            sliceSource.connect(sliceAnalyser);
            sliceAnalyser.connect(sliceOfflineCtx.destination);
            sliceSource.start(0);
            try {
                await sliceOfflineCtx.startRendering();
                sliceAnalyser.getByteFrequencyData(tempFreqDataArray);
                for (let f = 0; f < numRawFrequencyBins; f++) {
                    fullSongSpectrogramDataMatrix[t][f] = tempFreqDataArray[f] / 255.0;
                }
            } catch (renderError) {
                console.error(`Error rendering STFT slice ${t}:`, renderError);
                for (let f = 0; f < numRawFrequencyBins; f++) fullSongSpectrogramDataMatrix[t][f] = 0.0;
            }
        }

        drawFullSpectrogram(fullSongSpectrogramDataMatrix);
        spectrogramStatusDisplay.textContent = "Full spectrogram ready.";

    } catch (error) {
        console.error("Error fetching or processing full audio for spectrogram:", error);
        spectrogramStatusDisplay.textContent = `Spectrogram Error: ${error.message.substring(0,100)}`;
        originalFullSongAudioBuffer = null;
        fullSongSpectrogramDataMatrix = null;
        currentSongDuration = 0;
        clearFullSpectrogramDisplay();
    }
}

function drawFullSpectrogram(spectrogramMatrix) {
    if (!fullSpectrogramCtx || fullSpectrogramCanvas.width <= 0 || !spectrogramMatrix) return;
    clearFullSpectrogramDisplay();

    const numTimeSlices = spectrogramMatrix.length;
    if (numTimeSlices === 0) return;
    const numRawFrequencyBins = spectrogramMatrix[0].length; 
    if (numRawFrequencyBins === 0) return;

    const startBin = Math.floor(yMinFreqPercent * numRawFrequencyBins);
    const endBin = Math.ceil(yMaxFreqPercent * numRawFrequencyBins);
    const numVisibleBins = Math.max(1, endBin - startBin); 

    const cellWidth = fullSpectrogramCanvas.width / numTimeSlices;
    const cellHeight = fullSpectrogramCanvas.height / numVisibleBins;

    for (let t = 0; t < numTimeSlices; t++) {
        for (let binIdx = 0; binIdx < numVisibleBins; binIdx++) {
            const actualMatrixBin = startBin + binIdx;
            if (actualMatrixBin >= numRawFrequencyBins) continue; 

            const normalizedMagnitude = spectrogramMatrix[t][actualMatrixBin];
            fullSpectrogramCtx.fillStyle = mapNormalizedToViridis(normalizedMagnitude);
            fullSpectrogramCtx.fillRect(
                t * cellWidth,
                fullSpectrogramCanvas.height - ((binIdx + 1) * cellHeight),
                cellWidth,
                cellHeight
            );
        }
    }
}

// --- Audio Playback and Control ---
function updateSpectrogramFrequencyWindowControls() {
    const sampleRate = originalFullSongAudioBuffer ? originalFullSongAudioBuffer.sampleRate : (audioCtx ? audioCtx.sampleRate : DEFAULT_SR);
    const nyquist = sampleRate / 2;

    yMinFreqValueDisplay.textContent = Math.round(yMinFreqPercent * nyquist) + " Hz";
    yMaxFreqValueDisplay.textContent = Math.round(yMaxFreqPercent * nyquist) + " Hz";
}

function initFilterControls() {
    highpassSlider.value = DEFAULT_HIGHPASS_FREQ;
    highpassValueDisplay.textContent = DEFAULT_HIGHPASS_FREQ;
    lowpassSlider.value = DEFAULT_LOWPASS_FREQ;
    lowpassValueDisplay.textContent = DEFAULT_LOWPASS_FREQ;

    yMinFreqSlider.value = yMinFreqPercent * 100;
    yMaxFreqSlider.value = yMaxFreqPercent * 100;
    updateSpectrogramFrequencyWindowControls();


    const updateAudioFilter = (isHighpass, freq) => {
        const targetNode = isHighpass ? highpassFilterNode : lowpassFilterNode;
        if (targetNode && audioCtx) {
             targetNode.frequency.setValueAtTime(freq, audioCtx.currentTime);
        }
        if(isHighpass) highpassValueDisplay.textContent = freq;
        else lowpassValueDisplay.textContent = freq;

        if (originalFullSongAudioBuffer && currentSongFilePath) {
            console.log("Audio filter changed, regenerating full spectrogram STFT data.");
            if (window.filterChangeTimeout) clearTimeout(window.filterChangeTimeout);
            window.filterChangeTimeout = setTimeout(() => {
                fetchFullAudioAndGenerateSpectrogram(currentSongFilePath); 
            }, 300);
        }
    };

    highpassSlider.addEventListener('input', (event) => updateAudioFilter(true, parseFloat(event.target.value)));
    lowpassSlider.addEventListener('input', (event) => updateAudioFilter(false, parseFloat(event.target.value)));

    yMinFreqSlider.addEventListener('input', (event) => {
        let newMinPercent = parseFloat(event.target.value) / 100;
        if (newMinPercent >= yMaxFreqPercent) {
            newMinPercent = yMaxFreqPercent - 0.01; 
            yMinFreqSlider.value = newMinPercent * 100;
        }
        yMinFreqPercent = Math.max(0, newMinPercent);
        updateSpectrogramFrequencyWindowControls();
        if (fullSongSpectrogramDataMatrix) drawFullSpectrogram(fullSongSpectrogramDataMatrix);
    });

    yMaxFreqSlider.addEventListener('input', (event) => {
        let newMaxPercent = parseFloat(event.target.value) / 100;
        if (newMaxPercent <= yMinFreqPercent) {
            newMaxPercent = yMinFreqPercent + 0.01; 
            yMaxFreqSlider.value = newMaxPercent * 100;
        }
        yMaxFreqPercent = Math.min(1, newMaxPercent);
        updateSpectrogramFrequencyWindowControls();
        if (fullSongSpectrogramDataMatrix) drawFullSpectrogram(fullSongSpectrogramDataMatrix);
    });
}

function playAudioAndFadeIn() {
    const liveStreamSrc = '/stream?cachebust=' + new Date().getTime();

    if (audioCtx && !sourceNode) {
        try {
            sourceNode = audioCtx.createMediaElementSource(audio);
            highpassFilterNode = audioCtx.createBiquadFilter();
            highpassFilterNode.type = "highpass";
            highpassFilterNode.frequency.value = parseFloat(highpassSlider.value);
            highpassFilterNode.Q.value = 1;
            lowpassFilterNode = audioCtx.createBiquadFilter();
            lowpassFilterNode.type = "lowpass";
            lowpassFilterNode.frequency.value = parseFloat(lowpassSlider.value);
            lowpassFilterNode.Q.value = 1;
            analyser = audioCtx.createAnalyser();
            analyser.fftSize = 512;
            analyser.minDecibels = -90; analyser.maxDecibels = -10; analyser.smoothingTimeConstant = 0.8;
            dataArray = new Uint8Array(analyser.frequencyBinCount);
            sourceNode.connect(highpassFilterNode);
            highpassFilterNode.connect(lowpassFilterNode);
            lowpassFilterNode.connect(analyser);
            analyser.connect(audioCtx.destination);
            console.log("Web Audio API nodes for live stream created.");
        } catch (e) {
            console.error("Error creating Web Audio API nodes:", e);
            analyser = highpassFilterNode = lowpassFilterNode = sourceNode = null;
        }
    } else if (audioCtx && sourceNode) {
        if (highpassFilterNode) highpassFilterNode.frequency.setValueAtTime(parseFloat(highpassSlider.value), audioCtx.currentTime);
        if (lowpassFilterNode) lowpassFilterNode.frequency.setValueAtTime(parseFloat(lowpassSlider.value), audioCtx.currentTime);
    }

    audio.src = liveStreamSrc;
    audio.load(); 
    audio.volume = 0;
    
    audio.play().then(() => {
        playBtn.disabled = true; stopBtn.disabled = false;
        fadeAudio(1.0, FADE_DURATION_MS);

        fetchAndSetLatestSongName().then(() => { 
            if (liveAnimationId) cancelAnimationFrame(liveAnimationId);
            resizeCanvases(); 
            drawLiveSpectrogramAndPlayhead(); 
            console.log("Live spectrogram and playhead drawing started.");

            if (currentSongFilePath) {
                fetchFullAudioAndGenerateSpectrogram(currentSongFilePath);
            } else {
                 spectrogramStatusDisplay.textContent = "Song path not ready for full spectrogram.";
                 serverEffectiveSongDurationSec = 0;
                 serverEffectiveCrossfadeDurationSec = 0;
            }
        });

    }).catch(error => {
        console.error("Error playing audio:", error);
        songDisplay.textContent = "Error trying to play: " + error.message;
        playBtn.disabled = false; stopBtn.disabled = true;
        currentSongDuration = 0;
        serverEffectiveSongDurationSec = 0;
        serverEffectiveCrossfadeDurationSec = 0;
    });
}

playBtn.onclick = () => {
    clearExistingFade();
    if (!audioCtx) {
        try {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
            console.log("AudioContext created. Sample rate: " + audioCtx.sampleRate);
            updateSpectrogramFrequencyWindowControls(); 
        } catch (e) {
            console.error("Error creating AudioContext:", e);
            songDisplay.textContent = "AudioContext not supported."; return;
        }
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume().then(() => playAudioAndFadeIn())
        .catch(e => { songDisplay.textContent = "Could not start audio. Click Listen again."; });
    } else playAudioAndFadeIn();
};

stopBtn.onclick = () => {
    clearExistingFade();
    if (liveAnimationId) cancelAnimationFrame(liveAnimationId); liveAnimationId = null;
    if (canvasCtx && spectrogramCanvas.width > 0 && !appContentOverlay.classList.contains('settings-hidden')) {
        canvasCtx.fillStyle = '#000';
        canvasCtx.fillRect(0, 0, spectrogramCanvas.width, spectrogramCanvas.height);
    }
    clearFullSpectrogramDisplay(); clearPlayhead();
    spectrogramStatusDisplay.textContent = "";
    originalFullSongAudioBuffer = null; fullSongSpectrogramDataMatrix = null;
    currentSongDuration = 0;
    serverEffectiveSongDurationSec = 0;
    serverEffectiveCrossfadeDurationSec = 0;

    if (audio.src && !audio.paused) audio.pause();
    audio.volume = 1.0; audio.removeAttribute('src'); audio.load(); 
    stopBtn.disabled = true; playBtn.disabled = false;
    songDisplay.textContent = "Stopped. Press Listen to start.";
};

// --- SSE and App Initialization ---
const evtSource = new EventSource("/events");
evtSource.onmessage = function(event) {
    if (event.data === "new_song") {
        console.log("New song detected by server.");
        if (!audio.paused || playBtn.disabled) { 
            fadeAudio(0, FADE_DURATION_MS, () => {
                songDisplay.textContent = "New song loading...";
                if (liveAnimationId) cancelAnimationFrame(liveAnimationId);
                clearFullSpectrogramDisplay(); clearPlayhead(); spectrogramStatusDisplay.textContent = "";
                originalFullSongAudioBuffer = null; fullSongSpectrogramDataMatrix = null;
                currentSongDuration = 0;
                serverEffectiveSongDurationSec = 0; serverEffectiveCrossfadeDurationSec = 0;
                playAudioAndFadeIn();
            });
        } else { 
            fetchAndSetLatestSongName().then(() => {
                clearFullSpectrogramDisplay(); clearPlayhead();
                spectrogramStatusDisplay.textContent = "New song available. Press Listen.";
                originalFullSongAudioBuffer = null; fullSongSpectrogramDataMatrix = null;
                currentSongDuration = 0;
                if (currentSongFilePath) fetchFullAudioAndGenerateSpectrogram(currentSongFilePath);
            });
        }
    }
};
evtSource.onerror = function(err) { console.error("EventSource failed:", err); songDisplay.textContent = "Connection to server updates lost."; };

function initializeApp() {
    // ### NEW: Create and set up toggle settings button ###
    toggleSettingsBtn = document.createElement('button');
    toggleSettingsBtn.id = 'toggleSettingsBtn';
    toggleSettingsBtn.textContent = 'Hide Settings';
    
    // Insert the button after the H1 title
    if (h1Title && h1Title.parentNode === appContentOverlay) {
        h1Title.parentNode.insertBefore(toggleSettingsBtn, h1Title.nextSibling);
    } else { // Fallback: prepend to overlay if H1 is not found as expected
        appContentOverlay.prepend(toggleSettingsBtn);
    }

    toggleSettingsBtn.addEventListener('click', () => {
        appContentOverlay.classList.toggle('settings-hidden');
        const isHidden = appContentOverlay.classList.contains('settings-hidden');
        toggleSettingsBtn.textContent = isHidden ? 'Show Settings' : 'Hide Settings';
        resizeCanvases(); // Re-check canvas sizes/visibility
    });
    // ### END NEW ###

    resizeCanvases();
    initFilterControls();
    fetchAndSetLatestSongName().then(() => {
        if (currentSongFilePath) {
             fetchFullAudioAndGenerateSpectrogram(currentSongFilePath); 
        } else {
            clearFullSpectrogramDisplay(); clearPlayhead();
            spectrogramStatusDisplay.textContent = "No song loaded initially.";
        }
    });
    playBtn.disabled = false; stopBtn.disabled = true;
}

window.addEventListener('load', initializeApp);
window.addEventListener('resize', resizeCanvases);

evtSource.onopen = function() {
    console.log("EventSource connection opened.");
    updateSpectrogramFrequencyWindowControls();
    const wasPlaying = !audio.paused && playBtn.disabled; 
    fetchAndSetLatestSongName().then(() => {
        if (!wasPlaying && currentSongFilePath && !originalFullSongAudioBuffer) {
             fetchFullAudioAndGenerateSpectrogram(currentSongFilePath);
        } else if (!currentSongFilePath) {
            clearFullSpectrogramDisplay(); clearPlayhead();
            spectrogramStatusDisplay.textContent = "No song available yet after SSE reconnect.";
            currentSongDuration = 0;
            serverEffectiveSongDurationSec = 0;
            serverEffectiveCrossfadeDurationSec = 0;
        }
    });
};

</script>
</body>
</html>"""

# ───────────────────────── main ─────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser("LoRA Radio (Full BG Spectrogram with Y-Window)")
    ap.add_argument("-w","--watch-dir", default="lightning_logs", help="folder to watch for pred_*.flac files")
    ap.add_argument("-p","--port", type=int, default=8000)
    ap.add_argument("--crossfade", type=float, default=CROSSFADE_SEC, help="Server-side crossfade duration in seconds (0 to disable)")
    args = ap.parse_args()

    CROSSFADE_SEC = args.crossfade
    # CROSSFADE_FRAMES is calculated based on the final CROSSFADE_SEC
    CROSSFADE_FRAMES = int(SR * CROSSFADE_SEC) if CROSSFADE_SEC > 0 else 0
    log.info(f"Server-side crossfade duration set to {CROSSFADE_SEC} seconds ({CROSSFADE_FRAMES} frames).")

    resolved_watch_dir = Path(args.watch_dir).resolve()
    if not resolved_watch_dir.exists() or not resolved_watch_dir.is_dir():
        log.error(f"Watch directory {resolved_watch_dir} does not exist or is not a directory."); raise SystemExit(1)

    # ### MODIFIED ###: Populate playlist with (Path, frames) tuples
    temp_playlist_items = []
    for p_glob in resolved_watch_dir.glob("**/pred_*.flac"):
        if p_glob.is_file():
            try:
                # Attempt to get frame count for initial scan
                # Wait for stability before trying to read info
                if not wait_until_stable(p_glob):
                    log.warning(f"Initial scan: {p_glob.name} was not stable. Skipping.")
                    continue
                
                frames = 0
                initial_sf = None
                try:
                    initial_sf = sf.SoundFile(str(p_glob), 'r')
                    if initial_sf.samplerate != SR or initial_sf.channels != CH:
                        log.warning(f"Initial scan: {p_glob.name} has wrong SR/CH. Skipping.")
                        continue
                    if initial_sf.frames == 0:
                        log.warning(f"Initial scan: {p_glob.name} is empty. Skipping.")
                        continue
                    frames = initial_sf.frames
                except Exception as e:
                    log.warning(f"Initial scan: Could not read info for {p_glob.name}: {e}. Skipping.")
                    continue
                finally:
                    if initial_sf: initial_sf.close()

                relative_p = p_glob.relative_to(resolved_watch_dir)
                temp_playlist_items.append({'path': relative_p, 'frames': frames, 'mtime': p_glob.stat().st_mtime, 'name': p_glob.name})
            except ValueError: 
                log.warning(f"Initial scan: File {p_glob.name} found but could not be made relative to {resolved_watch_dir}. Skipping.")
            except Exception as e: # Catch other potential errors during initial scan file processing
                log.warning(f"Initial scan: Error processing file {p_glob.name}: {e}. Skipping.")


    # Sort based on mtime and name
    sorted_temp_items = sorted(
        temp_playlist_items, 
        key=lambda item: (item['mtime'], item['name'])
    )

    # Populate global playlist with (Path, int_frames)
    with pl_lock:
        for item in sorted_temp_items:
            playlist.append((item['path'], item['frames']))


    if playlist:
        log.info(f"🎧  Watching {resolved_watch_dir}  ({len(playlist)} preloaded, latest: {playlist[-1][0].name}, {playlist[-1][1]} frames)")
        file_ready.set()
    else: log.info(f"🎧  Watching {resolved_watch_dir}  (0 files preloaded)")

    watcher_thread = Thread(target=start_watcher, args=(resolved_watch_dir,), daemon=True); watcher_thread.start()

    log.info(f"🚀 LoRA Radio starting on http://0.0.0.0:{args.port}")
    try:
        app.run(host="0.0.0.0", port=args.port, threaded=True, use_reloader=False)
    except KeyboardInterrupt: log.info("Keyboard interrupt received. Shutting down...")
    finally:
        log.info("Setting stop event for all threads..."); stop_evt.set()
        with sse_lock:
            for q_item_close in sse_subscribers: q_item_close.put("__close__")
        if watcher_thread.is_alive():
            log.info("Waiting for watcher thread to join..."); watcher_thread.join(timeout=3)
            if watcher_thread.is_alive(): log.warning("Watcher thread did not join in time.")
        log.info("LoRA Radio has shut down.")