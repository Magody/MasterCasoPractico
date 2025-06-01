import os
import json
import tempfile
from flask import Blueprint, request, jsonify
from src.pipeline.chat_pipeline import text_pipeline, audio_pipeline

api_bp = Blueprint("api", __name__)

def _save_audio_upload():
    """
    - If form‐upload (multipart), pulls from `request.files['file']`
    - Else if raw audio in body, writes request.get_data() to temp
    Returns path to temp file.
    """
    # 1) multipart/form-data case
    if 'file' in request.files:
        file = request.files['file']
        suffix = os.path.splitext(file.filename)[1] or '.wav'
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        file.save(path)
        return path

    # 2) raw audio body (audio/* or application/octet-stream)
    data = request.get_data()
    # try to pick a suffix
    mt = request.mimetype or ''
    if 'wav' in mt:
        suffix = '.wav'
    elif 'mpeg' in mt or 'mp3' in mt:
        suffix = '.mp3'
    else:
        suffix = ''
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, data)
    os.close(fd)
    return path


@api_bp.route("/generate", methods=["POST"])
def generate():
    """
    Single endpoint for both text and audio.

    Query params:
      ?mode=text|audio

    Fallback logic:
      • multipart/form-data or audio/* body → audio
      • JSON with "text" field → text
      • else → 400
    """
    mode = (request.args.get("mode") or "").lower()
    history = None
    wav_path = None

    # Force AUDIO if requested
    if mode == "audio":
        audio_path = _save_audio_upload()
        # optional history passed as JSON string in form-data
        raw_hist = request.form.get("history")
        try:
            history = json.loads(raw_hist) if raw_hist else None
        except json.JSONDecodeError:
            history = None

        history, wav_path, updated_history = audio_pipeline(
            audio_file=audio_path,
            history=history
        )

    # Force TEXT if requested
    elif mode == "text":
        data = request.get_json(silent=True) or {}
        user_text = data.get("text")
        history   = data.get("history")

        if not user_text:
            return jsonify({"error": "`text` is required in JSON body"}), 400

        history, wav_path, updated_history = text_pipeline(
            user_text=user_text,
            history=history
        )

    # AUTO-DETECT
    else:
        # audio-style content?
        if request.files or (request.mimetype and request.mimetype.startswith("audio")):
            audio_path = _save_audio_upload()
            history = None
            history, wav_path, updated_history = audio_pipeline(
                audio_file=audio_path,
                history=history
            )
        else:
            data = request.get_json(silent=True) or {}
            user_text = data.get("text")
            if not user_text:
                return jsonify({"error": "No audio upload and no `text` in JSON"}), 400
            history = data.get("history")
            history, wav_path, updated_history = text_pipeline(
                user_text=user_text,
                history=history
            )

    # Build response
    reply = updated_history[-1][1] if updated_history else ""
    return jsonify({
        "reply":   reply,
        "history": updated_history
    })
