import torch
import gradio as gr
import yt_dlp as youtube_dl
from faster_whisper import WhisperModel
import tempfile
import os
import submitit
import time

FILE_LIMIT_MB = 1000
YT_LENGTH_LIMIT_S = 3600*4  # limit to 1 hour YouTube files

executor = submitit.AutoExecutor(folder="submitit_logs")
executor.update_parameters(timeout_min=30, slurm_partition="gpu", mem_gb=64, gpus_per_node=1)

def convert_to_vtt(segments):
    """
    Convert Faster-Whisper segments to VTT subtitle format.
    
    Args:
        segments: Generator of transcription segments from faster-whisper
        
    Returns:
        str: VTT formatted subtitles as a string
    """
    def format_timestamp(seconds):
        """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)"""
        if seconds is None:
            return "99:59:59.999"  # Use max time for None values
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds_remainder = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds_remainder:06.3f}".replace('.', ',')
    
    # Start with VTT header
    vtt_output = "WEBVTT\n\n"
    
    # Process each segment
    for i, segment in enumerate(segments, 1):
        # Format the subtitle entry
        vtt_output += f"{i}\n"
        vtt_output += f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
        vtt_output += f"{segment.text.strip()}\n\n"
    
    return vtt_output

def transcribe(file_path):
    print("Submitting transcription job")
    job = executor.submit(do_transcribe, file_path)
    result = job.result()    
    return result

def do_transcribe(file_path):
    # Specify model path - using the same path as in the original code
    MODEL_NAME = "/slurm-share/tanel/models/whisper-large-v3-turbo-et-subs-ct2"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "float32"
    
    print(f"Starting to transcribe file {file_path} using {MODEL_NAME} using device {device}")
    
    # Load the model with faster-whisper
    model = WhisperModel(MODEL_NAME, device=device, compute_type=compute_type)
    
    # Transcribe the audio file
    segments, info = model.transcribe(
        file_path,
        task="transcribe",
        language="et",
        beam_size=5,
        vad_filter=True,
        word_timestamps=False  # Set to True if you need word-level timestamps
    )
    
    # Convert segments to VTT format
    result = convert_to_vtt(segments)
    return result

def return_yt_html_embed(yt_url):
    video_id = yt_url.split("?v=")[-1]
    HTML_str = (
        f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str

def download_yt_audio(yt_url, filename):
    info_loader = youtube_dl.YoutubeDL()
    
    try:
        info = info_loader.extract_info(yt_url, download=False)
    except youtube_dl.utils.DownloadError as err:
        raise gr.Error(str(err))
    
    file_length = info["duration_string"]
    file_h_m_s = file_length.split(":")
    file_h_m_s = [int(sub_length) for sub_length in file_h_m_s]
    
    if len(file_h_m_s) == 1:
        file_h_m_s.insert(0, 0)
    if len(file_h_m_s) == 2:
        file_h_m_s.insert(0, 0)
    file_length_s = file_h_m_s[0] * 3600 + file_h_m_s[1] * 60 + file_h_m_s[2]
    
    if file_length_s > YT_LENGTH_LIMIT_S:
        yt_length_limit_hms = time.strftime("%HH:%MM:%SS", time.gmtime(YT_LENGTH_LIMIT_S))
        file_length_hms = time.strftime("%HH:%MM:%SS", time.gmtime(file_length_s))
        raise gr.Error(f"Maximum YouTube length is {yt_length_limit_hms}, got {file_length_hms} YouTube video.")
    
    ydl_opts = {"outtmpl": filename, "format": "worstvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best", "extract_audio": True}
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([yt_url])
        except youtube_dl.utils.ExtractorError as err:
            raise gr.Error(str(err))

def yt_transcribe(yt_url, max_filesize=75.0):
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = os.path.join(tmpdirname, "video.mp4")
        download_yt_audio(yt_url, filepath)
        text = transcribe(filepath)
        
    return text

demo = gr.Blocks(theme=gr.themes.Ocean())

mf_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources="microphone", type="filepath")
    ],
    outputs=gr.Textbox(label="VTT subtiitrid", elem_id="text", show_label=True, show_copy_button=True, autoscroll=False, interactive=True),
    title="Genereeri eestikeelsed subtiitrid!",
    description=(
        "Siin saad genereerida eestikeelsed subtiitrid mikrofoniga salvestatud kõnelõigule!"
    ),
    allow_flagging="never",
)

file_transcribe = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(sources="upload", type="filepath", label="Audio file")
    ],
    outputs=gr.Textbox(label="VTT subtiitrid", elem_id="text", show_label=True, show_copy_button=True, autoscroll=False, interactive=True),
    title="Genereeri eestikeelsed subtiitrid!",
    description=(
        "Siin saad genereerida eestikeelsed subtiitrid üleslaetud kõnesalvestusele!"
    ),
    allow_flagging="never",
)

yt_transcribe = gr.Interface(
    fn=yt_transcribe,
    inputs=[
        gr.Textbox(lines=1, placeholder="Sisesta siia YouTube video URL", label="YouTube URL")
    ],
    outputs=gr.Textbox(label="VTT subtiitrid", elem_id="text", show_label=True, show_copy_button=True, autoscroll=False, interactive=True),
    title="Genereeri eestikeelsed subtiitrid!",
    description=(
        "Siin saad genereerida eestikeelsed subtiitrid YouTube videole!"
    ),
    allow_flagging="never",
)

with demo:
    gr.TabbedInterface([mf_transcribe, file_transcribe, yt_transcribe], ["Mikrofon", "Helifail", "YouTube"])

demo.launch(server_port=7862, root_path="/subtitreeri")
