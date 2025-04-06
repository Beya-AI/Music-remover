import streamlit as st
import os
import subprocess
import sys
import shutil
import time
from pathlib import Path
import tempfile # For handling uploaded files safely

# --- Configuration ---
# Use session state for persistence across reruns
if 'SESSION_ID' not in st.session_state: # Simple way to manage temp folders per session
    st.session_state.SESSION_ID = str(int(time.time() * 1000))

# Define base temporary directory based on environment
if 'BASE_TEMP_DIR' not in st.session_state:
    if os.environ.get("SPACE_ID"):
        st.session_state.BASE_TEMP_DIR = Path(f"/tmp/st_demucs_{st.session_state.SESSION_ID}")
    else:
        st.session_state.BASE_TEMP_DIR = Path(f"st_demucs_output_{st.session_state.SESSION_ID}")

if 'DEMUCS_MODEL_NAME' not in st.session_state:
    st.session_state.DEMUCS_MODEL_NAME = "htdemucs_ft"

# Ensure base temp dir exists
st.session_state.BASE_TEMP_DIR.mkdir(parents=True, exist_ok=True)
print(f"Using temporary directory: {st.session_state.BASE_TEMP_DIR}")


# --- Helper: run_command (same robust version) ---
def run_command(command, description="command", status_placeholder=None):
    """Runs a shell command, yields output lines for streaming, and checks status."""
    full_output = ""
    cmd_str = ' '.join(command)
    print(f"\n--- Running {description}: {cmd_str} ---")
    log_line = f"Running: {description}..."
    if status_placeholder: status_placeholder.write(log_line)
    else: print(log_line)

    try:
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = "1"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='ignore', env=env)

        if process.stdout:
             while True:
                 line = process.stdout.readline()
                 if not line: break
                 line_strip = line.strip()
                 print(line_strip) # Log to console
                 full_output += line
                 # Update status selectively
                 if status_placeholder and ("%" in line_strip or line_strip.startswith("Separating") or line_strip.startswith("Downloading")):
                      try: status_placeholder.write(f"Progress: {line_strip}")
                      except Exception: pass # Ignore errors if placeholder is gone

        process.wait()
        final_log_line = f"{description} finished."
        if process.returncode == 0:
            print(f"--- {description} finished successfully. ---")
            if status_placeholder: status_placeholder.write(final_log_line + " Success.")
            return True, full_output
        else:
            error_log = f"\n--- ERROR during {description} (Return Code: {process.returncode}) ---\nCommand: {cmd_str}\nOutput:\n{full_output}\n--- {description} failed. ---"
            print(error_log)
            if status_placeholder: status_placeholder.error(f"{description} failed. Check logs.")
            return False, full_output
    except Exception as e:
        error_msg = f"--- An unexpected ERROR occurred running {description}: {e} ---"
        print(error_msg)
        import traceback; traceback.print_exc()
        if status_placeholder: status_placeholder.error(f"Unexpected error: {e}")
        return False, error_msg


# --- Helper: Download Video (New) ---
def download_video_yt_dlp(url, output_dir, status_placeholder):
    os.makedirs(output_dir, exist_ok=True)
    # Let yt-dlp determine the filename, usually includes title/id
    out_template = str(Path(output_dir) / '%(title)s [%(id)s].%(ext)s')
    final_video_path = None # We'll find the path after download

    status_placeholder.write(f"Attempting video download for {url}...")
    ydl_cmd = [
        'yt-dlp', '--no-check-certificate', '--no-warnings', '--progress',
        '-f', 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best', # Common format choice
        '--merge-output-format', 'mp4', # Ensure container is mp4
        '-o', out_template,
        url
    ]

    success, output = run_command(ydl_cmd, "yt-dlp Video Download", status_placeholder)

    if not success:
        status_placeholder.error("Video download failed.")
        return None

    # Find the downloaded file (most recently created mp4 in the dir)
    try:
        downloaded_files = sorted(Path(output_dir).glob('*.mp4'), key=os.path.getmtime, reverse=True)
        if downloaded_files:
            final_video_path = str(downloaded_files[0])
            status_placeholder.success(f"Video download successful: {Path(final_video_path).name}")
            return final_video_path
        else:
             status_placeholder.error("yt-dlp ran but no MP4 video file found.")
             return None
    except Exception as e:
         status_placeholder.error(f"Error finding downloaded video file: {e}")
         return None


# --- Helper: Download Audio (Modified) ---
def download_audio_yt_dlp(url, output_dir, audio_filename, status_placeholder):
    # (Similar logic as before, returns path on success)
    os.makedirs(output_dir, exist_ok=True)
    final_audio_path = os.path.join(output_dir, audio_filename)
    temp_out_template = os.path.join(output_dir, 'temp_audio_%(id)s.%(ext)s')

    # Cleanup previous files
    for old_temp in Path(output_dir).glob('temp_audio*.wav'):
        try: os.remove(old_temp)
        except OSError: pass
    if os.path.exists(final_audio_path):
        try: os.remove(final_audio_path)
        except OSError: pass

    status_placeholder.write(f"Attempting audio download for {url}...")
    ydl_cmd = [
        'yt-dlp', '--no-check-certificate', '--no-warnings', '--progress',
        '-f', 'bestaudio/best', '-x', '--audio-format', 'wav', '--audio-quality', '0',
        '-o', temp_out_template, url
    ]
    success, output = run_command(ydl_cmd, "yt-dlp Audio Download", status_placeholder)
    if not success:
        status_placeholder.error("Audio download failed.")
        return None

    downloaded_files = list(Path(output_dir).glob('temp_audio*.wav'))
    if not downloaded_files:
        status_placeholder.error("yt-dlp ran but no WAV audio file found.")
        return None

    actual_temp_path = str(downloaded_files[0])
    try:
        shutil.move(actual_temp_path, final_audio_path)
        if os.path.exists(final_audio_path):
             status_placeholder.success("Audio download successful.")
             return final_audio_path
        else: raise OSError("File not found after move")
    except OSError as e:
        status_placeholder.error(f"Error moving/finding downloaded audio: {e}")
        return None


# --- Helper: Convert to WAV (New) ---
def convert_to_wav(input_path, output_dir, output_filename="converted.wav", status_placeholder=None):
    """Converts an audio file to WAV using ffmpeg."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / output_filename
    if output_path.exists():
        try: os.remove(output_path)
        except OSError: pass

    command = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error',
        '-i', str(input_path),   # Input file
        '-acodec', 'pcm_s16le', # Standard WAV codec
        '-ac', '2',            # Stereo channels
        '-ar', '44100',        # Sample rate Demucs likes
        '-y',                  # Overwrite output
        str(output_path)
    ]
    success, output = run_command(command, "ffmpeg Conversion to WAV", status_placeholder)
    if success and output_path.exists() and output_path.stat().st_size > 100:
        if status_placeholder: status_placeholder.write("Conversion to WAV successful.")
        return str(output_path)
    else:
        if status_placeholder: status_placeholder.error(f"ffmpeg conversion failed. Output:\n{output}")
        return None

# --- Helper: Demucs Separation (Mostly unchanged) ---
def separate_with_demucs(input_audio_path, output_dir, model_name, status_placeholder):
    if not input_audio_path or not os.path.exists(input_audio_path):
        if status_placeholder: status_placeholder.error(f"Input audio for Demucs not found: {input_audio_path}")
        return False, None

    os.makedirs(output_dir, exist_ok=True)
    status_placeholder.write(f"Starting Demucs Separation (Model: {model_name})... (This can take minutes!)")

    demucs_cmd = ['demucs', '-o', str(output_dir), '-n', model_name, str(input_audio_path)]
    success, output = run_command(demucs_cmd, "Demucs Separation", status_placeholder)

    if not success:
        status_placeholder.error(f"Demucs separation failed.")
        return False, None

    input_filename_base = Path(input_audio_path).stem
    final_stems_path = Path(output_dir) / model_name / input_filename_base

    if not final_stems_path.is_dir():
         status_placeholder.error(f"Expected Demucs output directory not found: {final_stems_path}")
         return False, str(final_stems_path)

    status_placeholder.success("Demucs separation finished.")
    return True, str(final_stems_path)

# --- Streamlit UI ---

st.set_page_config(layout="wide")
st.title("🎬🎶 Audio/Video Downloader & Vocal Separator")
st.markdown("""
Choose your input source (YouTube URL or Local File) and select the desired output.
**Note:** Processing vocals requires significant computation (GPU recommended for speed).
""")

# Initialize session state
if 'processing' not in st.session_state: st.session_state.processing = False
if 'final_output_path' not in st.session_state: st.session_state.final_output_path = None
if 'final_output_name' not in st.session_state: st.session_state.final_output_name = None
if 'final_output_mime' not in st.session_state: st.session_state.final_output_mime = None
if 'error_message' not in st.session_state: st.session_state.error_message = None

# --- Input Selection ---
input_method = st.radio("Choose Input Method:", ("YouTube URL", "Upload Local File"), horizontal=True)

url_input = ""
uploaded_file = None

if input_method == "YouTube URL":
    url_input = st.text_input("Enter YouTube URL:", placeholder="https://www.youtube.com/watch?v=...")
else:
    uploaded_file = st.file_uploader("Upload Audio or Video File:", type=['mp3', 'wav', 'm4a', 'mp4', 'mov', 'ogg', 'flac']) # Add relevant types

# --- Output Selection ---
output_options = ["Download Video", "Download Audio", "Download Vocals Only"]
# Disable video download if input is file upload
if input_method == "Upload Local File":
    output_options.remove("Download Video")

output_choice = st.radio("Select Desired Output:", output_options, horizontal=True)

# --- Processing Button ---
submit_button = st.button("Process Request", disabled=st.session_state.processing)

# --- Status & Results Area ---
status_placeholder = st.empty()
result_placeholder = st.empty()

if submit_button:
    # --- Validation ---
    source_valid = False
    if input_method == "YouTube URL" and url_input and url_input.strip().startswith(('http:', 'https:')):
        source_valid = True
        source_description = url_input
    elif input_method == "Upload Local File" and uploaded_file is not None:
        source_valid = True
        source_description = uploaded_file.name
    else:
        st.warning("Please provide a valid YouTube URL or upload a file.")

    if source_valid:
        # Reset state and start processing
        st.session_state.processing = True
        st.session_state.final_output_path = None
        st.session_state.final_output_name = None
        st.session_state.final_output_mime = None
        st.session_state.error_message = None
        status_placeholder.empty()
        result_placeholder.empty()

        # Use st.status for progress updates
        with st.status("Processing request...", expanded=True) as status_ctx:
            try:
                start_time = time.time()
                current_step = "Starting..."
                # Define unique paths for this run using session ID
                run_temp_dir = st.session_state.BASE_TEMP_DIR
                run_temp_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists
                print(f"Using run directory: {run_temp_dir}")

                temp_input_file = None # Path to file saved from upload
                source_audio_wav_path = None # Path to WAV for demucs/audio output
                demucs_output_dir = run_temp_dir / "demucs_stems"

                # --- Handle Input Type ---
                if input_method == "Upload Local File":
                    status_ctx.write(f"Handling uploaded file: {uploaded_file.name}")
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix, dir=run_temp_dir) as tmp_f:
                        tmp_f.write(uploaded_file.getvalue())
                        temp_input_file = tmp_f.name
                        print(f"Saved uploaded file to: {temp_input_file}")

                    # If output is Audio or Vocals, ensure we have WAV
                    if output_choice in ["Download Audio", "Download Vocals Only"]:
                        if Path(temp_input_file).suffix.lower() == ".wav":
                            source_audio_wav_path = temp_input_file # Use directly if WAV
                            status_ctx.write("Uploaded file is WAV.")
                        else:
                            # Convert to WAV
                            status_ctx.write("Converting uploaded file to WAV...")
                            source_audio_wav_path = convert_to_wav(temp_input_file, run_temp_dir, "converted_upload.wav", status_ctx)
                            if not source_audio_wav_path: raise ValueError("Failed to convert uploaded file to WAV.")

                # --- Process based on Output Choice ---
                if output_choice == "Download Video":
                    if input_method == "YouTube URL":
                        current_step = "Downloading Video"
                        status_ctx.write("Downloading video...")
                        video_path = download_video_yt_dlp(url_input, str(run_temp_dir), status_ctx)
                        if not video_path: raise ValueError("Video download failed.")
                        st.session_state.final_output_path = video_path
                        st.session_state.final_output_name = Path(video_path).name
                        st.session_state.final_output_mime = "video/mp4"
                    else: # Should not happen due to UI logic, but good practice
                        raise ValueError("Video download only available for YouTube URLs.")

                elif output_choice == "Download Audio":
                    if input_method == "YouTube URL":
                        current_step = "Downloading Audio"
                        status_ctx.write("Downloading audio...")
                        audio_path = download_audio_yt_dlp(url_input, str(run_temp_dir), "downloaded_audio.wav", status_ctx)
                        if not audio_path: raise ValueError("Audio download failed.")
                        st.session_state.final_output_path = audio_path
                        st.session_state.final_output_name = "original_audio.wav"
                        st.session_state.final_output_mime = "audio/wav"
                    elif input_method == "Upload Local File":
                        current_step = "Preparing Uploaded Audio"
                        # source_audio_wav_path was prepared earlier (or is the original if already WAV)
                        if not source_audio_wav_path: raise ValueError("Could not prepare uploaded audio.")
                        st.session_state.final_output_path = source_audio_wav_path
                        st.session_state.final_output_name = f"original_{Path(uploaded_file.name).stem}.wav" # Name based on original
                        st.session_state.final_output_mime = "audio/wav"

                elif output_choice == "Download Vocals Only":
                    # Step 1: Ensure we have source WAV audio
                    if input_method == "YouTube URL":
                        current_step = "Downloading Audio for Separation"
                        status_ctx.write("Downloading audio for separation...")
                        source_audio_wav_path = download_audio_yt_dlp(url_input, str(run_temp_dir), "source_for_demucs.wav", status_ctx)
                    # For upload, source_audio_wav_path was prepared earlier
                    if not source_audio_wav_path: raise ValueError("Failed to get source WAV audio for separation.")

                    # Step 2: Run Demucs
                    current_step = "Separating Vocals"
                    status_ctx.write("Running Demucs...")
                    sep_success, final_stems_dir = separate_with_demucs(
                        source_audio_wav_path,
                        str(demucs_output_dir),
                        st.session_state.DEMUCS_MODEL_NAME,
                        status_ctx
                    )
                    if not sep_success: raise ValueError("Demucs separation failed.")

                    # Step 3: Locate Vocals file
                    vocals_path = Path(final_stems_dir) / "vocals.wav"
                    if vocals_path.exists() and vocals_path.stat().st_size > 100:
                        st.session_state.final_output_path = str(vocals_path)
                        st.session_state.final_output_name = f"vocals_{Path(source_audio_wav_path).stem}.wav"
                        st.session_state.final_output_mime = "audio/wav"
                    else:
                        raise ValueError(f"Vocals.wav not found or empty after separation in {final_stems_dir}")

                # --- Success ---
                total_time = time.time() - start_time
                status_ctx.update(label=f"Processing Complete ({total_time:.2f}s)", state="complete", expanded=False)
                st.session_state.error_message = None

            except Exception as e:
                st.session_state.error_message = f"Error during {current_step}: {e}"
                status_ctx.update(label=f"Error: {e}", state="error", expanded=True)
                print(f"Error during {current_step}: {e}") # Log error
                import traceback; traceback.print_exc() # Log full traceback

            finally:
                # Clean up temporary uploaded file if it exists
                if temp_input_file and os.path.exists(temp_input_file):
                    # Add small delay before removing? Might not be needed.
                    # time.sleep(0.5)
                    try:
                        # If the temp file *is* the final output (e.g. uploaded WAV for 'Download Audio'), don't delete it yet
                        if temp_input_file != st.session_state.final_output_path:
                             os.remove(temp_input_file)
                             print(f"Cleaned up temp input: {temp_input_file}")
                    except OSError as e_clean:
                         print(f"Warning: Failed to clean up temp input {temp_input_file}: {e_clean}")
                # Note: Demucs output and other temp files within run_temp_dir are harder to clean selectively
                # For simplicity here, we might leave them, or delete the whole run_temp_dir later if needed.
                st.session_state.processing = False # Re-enable button

# --- Display Final Result Area ---
if st.session_state.error_message:
    result_placeholder.error(st.session_state.error_message)
elif st.session_state.final_output_path and Path(st.session_state.final_output_path).exists():
    result_placeholder.success(f"Your requested file '{st.session_state.final_output_name}' is ready!")

    final_path = Path(st.session_state.final_output_path)
    # Show audio player for audio types
    if st.session_state.final_output_mime.startswith("audio/"):
        try:
            audio_bytes = final_path.read_bytes()
            result_placeholder.audio(audio_bytes, format=st.session_state.final_output_mime)
        except Exception as e_audio:
            result_placeholder.warning(f"Could not display audio preview: {e_audio}")

    # Show download button
    try:
        with open(final_path, "rb") as fp:
             result_placeholder.download_button(
                 label=f"Download: {st.session_state.final_output_name}",
                 data=fp,
                 file_name=st.session_state.final_output_name,
                 mime=st.session_state.final_output_mime,
                 key=f"dl_final_{st.session_state.SESSION_ID}" # Ensure key changes if path changes
             )
    except Exception as e_dl:
         result_placeholder.error(f"Error preparing download button: {e_dl}")

elif not st.session_state.processing and submit_button: # If process finished but no output path
     result_placeholder.warning("Processing finished, but no output file was generated. Check status messages and logs.")
