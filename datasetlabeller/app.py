import json
import os
from flask import Flask, render_template, request, jsonify, url_for
import librosa

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24) # For session security, though not used heavily here
app.config['TRACK_INFO_FILE'] = 'trackinfo.json'
app.config['STATIC_FOLDER'] = 'static'

TRACK_INFO_DATA = {}
ALL_TAGS_MASTER = set()
ALL_INSTRUMENTS_MASTER = set()

def load_track_info_from_file():
    """Loads track info from the JSON file and populates master lists."""
    global TRACK_INFO_DATA, ALL_TAGS_MASTER, ALL_INSTRUMENTS_MASTER
    try:
        with open(app.config['TRACK_INFO_FILE'], 'r', encoding='utf-8') as f:
            TRACK_INFO_DATA = json.load(f)
        
        ALL_TAGS_MASTER.clear()
        ALL_INSTRUMENTS_MASTER.clear()
        for track_details in TRACK_INFO_DATA.values():
            ALL_TAGS_MASTER.update(tag.strip() for tag in track_details.get("tags", []) if tag.strip())
            ALL_INSTRUMENTS_MASTER.update(inst.strip() for inst in track_details.get("instruments", []) if inst.strip())
        app.logger.info(f"Successfully loaded {app.config['TRACK_INFO_FILE']}.")
        app.logger.info(f"Found {len(ALL_TAGS_MASTER)} unique tags and {len(ALL_INSTRUMENTS_MASTER)} unique instruments.")

    except FileNotFoundError:
        TRACK_INFO_DATA = {}
        app.logger.warning(f"{app.config['TRACK_INFO_FILE']} not found. Starting with empty data. Please create it.")
    except json.JSONDecodeError as e:
        TRACK_INFO_DATA = {}
        app.logger.error(f"Error decoding {app.config['TRACK_INFO_FILE']}: {e}. Starting with empty data.")
    except Exception as e:
        TRACK_INFO_DATA = {}
        app.logger.error(f"An unexpected error occurred while loading track info: {e}")


def save_track_info_to_file():
    """Saves the current TRACK_INFO_DATA to the JSON file."""
    try:
        # Ensure the directory exists if the path is nested (not the case here but good practice)
        # os.makedirs(os.path.dirname(app.config['TRACK_INFO_FILE']) or '.', exist_ok=True)
        with open(app.config['TRACK_INFO_FILE'], 'w', encoding='utf-8') as f:
            json.dump(TRACK_INFO_DATA, f, indent=4, ensure_ascii=False) # ensure_ascii=False for wider char support
        load_track_info_from_file() # Reload to update master lists and ensure consistency
        app.logger.info(f"Successfully saved data to {app.config['TRACK_INFO_FILE']}.")
        return True
    except Exception as e:
        app.logger.error(f"Error saving {app.config['TRACK_INFO_FILE']}: {e}")
        return False

@app.before_request
def initial_load():
    # This ensures data is loaded before every request if it hasn't been loaded yet.
    # For a simple app, loading once at startup might be enough, but this is robust.
    if not TRACK_INFO_DATA: # Check if it's empty
        load_track_info_from_file()

@app.route('/')
def index():
    """Serves the main page."""
    # Sort track keys: Album1 first, then Album2, then alphabetically within each
    def sort_key_func(item):
        parts = item.split('/')
        game_prefix = parts[0]
        numeric_part = ''.join(filter(str.isdigit, parts[1]))
        return (game_prefix, int(numeric_part) if numeric_part else 0, item)

    sorted_track_keys = sorted(TRACK_INFO_DATA.keys(), key=sort_key_func)
    
    return render_template('index.html',
                           track_info_dict=TRACK_INFO_DATA, # Pass the whole dict if needed by template
                           sorted_track_keys=sorted_track_keys,
                           all_tags_master=sorted(list(ALL_TAGS_MASTER)),
                           all_instruments_master=sorted(list(ALL_INSTRUMENTS_MASTER)))

@app.route('/get_track_details/<path:track_key>')
def get_track_details(track_key):
    """Returns details for a specific track."""
    if track_key in TRACK_INFO_DATA:
        track_data = TRACK_INFO_DATA[track_key]
        # Construct the audio URL using url_for for robustness
        audio_url = url_for('static', filename=track_key, _external=False) # _external=False for relative path
        
        return jsonify({
            "status": "success",
            "data": track_data,
            "audio_url": audio_url,
            "key": track_key
        })
    return jsonify({"status": "error", "message": "Track not found"}), 404

@app.route('/save_track_details', methods=['POST'])
def save_track_details():
    """Saves the updated details for a track."""
    global TRACK_INFO_DATA # To modify the global dictionary
    try:
        payload = request.get_json()
        track_key = payload.get('key')
        updated_details = payload.get('data')

        if not track_key or updated_details is None:
            return jsonify({"status": "error", "message": "Missing track key or data in payload"}), 400

        if track_key not in TRACK_INFO_DATA:
            return jsonify({"status": "error", "message": f"Track key '{track_key}' not found."}), 404

        # Safely get and process each field
        game_val = updated_details.get("game")
        boss_val = updated_details.get("boss")
        stage_val = updated_details.get("stage")
        description_val = updated_details.get("description")
        lyrics_val = updated_details.get("lyrics")  

        tags_list = updated_details.get("tags", []) # Default to empty list if key is missing
        instruments_list = updated_details.get("instruments", []) # Default to empty list

        bpm_val        = updated_details.get("bpm")          # int or str -> we'll coerce
        music_key_val  = updated_details.get("key")          # musical key, e.g. "G minor"


        TRACK_INFO_DATA[track_key] = {
            "game":        (game_val or "").strip(),
            "boss":        (boss_val or "").strip() or None,
            "stage":       (stage_val or "").strip(),
            "description": (description_val or "").strip(),

            # NEW FIELDS ↓↓↓
            "bpm":   int(bpm_val) if str(bpm_val).strip().isdigit() else None,
            "key":   (music_key_val or "").strip(),
            "lyrics":   (lyrics_val or "").rstrip(),      # <-- add lyrics here


            "tags":        sorted(list({(tag or "").strip() for tag in tags_list})),
            "instruments": sorted(list({(inst or "").strip() for inst in instruments_list}))
        }
        
        if save_track_info_to_file():
            return jsonify({"status": "success", "message": "Track details saved successfully!"})
        else:
            return jsonify({"status": "error", "message": "Failed to write updated data to file."}), 500

    except Exception as e:
        app.logger.error(f"Error in save_track_details: {e}")
        # It's often good to log the full traceback in debug scenarios
        import traceback
        app.logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": f"An server error occurred: {str(e)}"}), 500

@app.route('/detect_bpm/<path:track_key>')
def detect_bpm(track_key):
    # build path under your static folder
    file_path = os.path.join(app.config['STATIC_FOLDER'], track_key)
    if not os.path.isfile(file_path):
        return jsonify({"status":"error","message":"File not found"}), 404
    try:
        # load audio (mono) at its native sampling rate
        y, sr = librosa.load(file_path, sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return jsonify({"status":"success", "bpm": float(tempo)})
    except Exception as e:
        app.logger.error(f"BPM detection failed for {track_key}: {e}")
        return jsonify({"status":"error","message": str(e)}), 500


if __name__ == '__main__':
    # Create trackinfo.json with initial data if it doesn't exist
    # This is helpful for first-time setup if you don't want to manually create the file.
    if not os.path.exists(app.config['TRACK_INFO_FILE']):
        app.logger.warning(f"{app.config['TRACK_INFO_FILE']} not found. Please create it with your track data.")
        # You could optionally create an empty one or one with a sample structure:
        # with open(app.config['TRACK_INFO_FILE'], 'w', encoding='utf-8') as f:
        #     json.dump({}, f)
        # app.logger.info(f"Created an empty {app.config['TRACK_INFO_FILE']}.")

    load_track_info_from_file() # Initial load on startup
    app.run(host='0.0.0.0',use_reloader=False) # debug=True for development, host for accessibility