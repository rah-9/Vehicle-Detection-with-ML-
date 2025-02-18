import cv2
import numpy as np
import gradio as gr
import torch
import yt_dlp
from ultralytics import YOLO

# -----------------------------
# Load and Configure YOLO Model
# -----------------------------
model = YOLO("yolov5n6u.pt")  # Using the updated YOLOv5 'u' model
if torch.cuda.is_available():
    model.to("cuda")

# Low-resolution size used for inference (for speed)
LOW_RES = (320, 180)

def detect_and_draw(frame):
    """
    Detect objects on a low-res copy of the frame,
    then scale and draw the detections on the original frame.
    """
    low_res_frame = cv2.resize(frame, LOW_RES)
    results = model(low_res_frame, verbose=False)
    
    # Calculate scaling factors to map low-res detections to the original frame
    scale_x = frame.shape[1] / LOW_RES[0]
    scale_y = frame.shape[0] / LOW_RES[1]

    # Draw each detection on the original frame
    for detection in results[0].boxes.data:
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
        label = f"{results[0].names[int(cls)]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

def error_frame(message, width=600, height=200):
    """
    Create a blank image with an overlaid error message.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(frame, message, (10, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return frame

def get_youtube_stream_url(youtube_url):
    """
    Use yt-dlp to extract a direct MP4 stream URL from a YouTube link.
    """
    ydl_opts = {'format': 'best[ext=mp4]', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        video_url = info.get('url', None)
        if not video_url:
            raise Exception("No suitable stream found.")
        return video_url

def process_video(video_path):
    """
    Process a local video file and yield processed frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield error_frame("Error opening video file")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = detect_and_draw(frame)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            yield result_rgb
        except Exception as e:
            yield error_frame("Processing error: " + str(e))
    cap.release()

def process_stream(stream_url):
    """
    Process a video stream (from YouTube or other live sources) and yield processed frames.
    """
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        yield error_frame("Error: Unable to open video stream.")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = detect_and_draw(frame)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            yield result_rgb
        except Exception as e:
            yield error_frame("Processing error: " + str(e))
    cap.release()

def process_input(source_type, youtube_url, video_file):
    """
    Depending on the selected source type, process either a YouTube video or a local file.
    Always yields a sequence of image frames.
    """
    if source_type == "YouTube":
        try:
            stream_url = get_youtube_stream_url(youtube_url)
            yield from process_stream(stream_url)
        except Exception as e:
            yield error_frame(f"Error fetching YouTube stream: {str(e)}")
    elif source_type == "Local File" and video_file is not None:
        yield from process_video(video_file)
    else:
        yield error_frame("No valid input provided.")

# -----------------------------
# Gradio Interface Setup
# -----------------------------
iface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Radio(choices=["YouTube", "Local File"], label="Select Source", value="YouTube"),
        gr.Textbox(label="YouTube URL", placeholder="Enter YouTube video URL here..."),
        gr.File(label="Upload Local Video", file_types=[".mp4", ".avi", ".mov"])
    ],
    outputs=gr.Image(type="numpy"),
    live=True,
    title="Real-time Object Detection with YOLOv5n6u",
    description=(
        "Select a source: YouTube or Local File. "
        "The model runs on a low-resolution copy for fast inference and draws detections on the original frame."
    )
)

if __name__ == "__main__":
    # Set share=True if you need a public URL (e.g., for research demos)
    iface.launch(share=True)
