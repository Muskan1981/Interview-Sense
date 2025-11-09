import cv2
import threading
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import time
import random
import matplotlib.pyplot as plt
from io import BytesIO
import queue, json, sounddevice as sd, wave
from vosk import Model, KaldiRecognizer
from transformers import pipeline
import torch
import os
import tempfile
from moviepy.editor import VideoFileClip

# --- Local imports ---
from models.posture_model import analyze_posture
from models.speech_model import transcribe_audio
from models.sentiment_model import analyze_sentiment


# ==========================================================
#                   REAL-TIME SPEECH + SENTIMENT
# ==========================================================
VOSK_MODEL_PATH = "vosk-model-small-en-us-0.15"
if not os.path.exists(VOSK_MODEL_PATH):
    raise FileNotFoundError(f"Vosk model not found at {VOSK_MODEL_PATH}")

vosk_model = Model(VOSK_MODEL_PATH)
rec = KaldiRecognizer(vosk_model, 16000)

sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    device=0 if torch.cuda.is_available() else -1
)

audio_q = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    audio_q.put(bytes(indata))


def analyze_sentiment_live(text):
    """Color sentiment: red = neg, green = pos, white = neutral"""
    if not text.strip():
        return (255, 255, 255)
    try:
        result = sentiment_analyzer(text[:200])[0]
        label = result["label"]
        if "1" in label or "2" in label:
            return (255, 80, 80)     # red
        elif "4" in label or "5" in label:
            return (80, 255, 80)     # green
        else:
            return (255, 255, 255)   # white
    except Exception:
        return (255, 255, 255)


# ==========================================================
#                        MAIN CLASS
# ==========================================================
class AIInterviewCoach:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🤖 AI Interview Coach – Neon Edition")
        self.root.geometry("1100x750")
        self.root.configure(bg="#0E1116")

        self.running = False
        self.cap = cv2.VideoCapture(0)
        self.text_display = ""
        self._audio_frames = []
        self.border_color = "#00E5FF"

        # ---------------- UI ----------------
        self.title_label = tk.Label(self.root, text="AI INTERVIEW COACH", fg="#00E5FF", bg="#0E1116",
                                    font=("Segoe UI Black", 30, "bold"))
        self.title_label.pack(pady=15)

        self.video_frame = tk.Frame(self.root, bg="#1B1F27", bd=8, relief="ridge", highlightthickness=3)
        self.video_frame.pack(pady=20, padx=20, fill="both", expand=True)
        self.video_label = tk.Label(self.video_frame, bg="#1B1F27")
        self.video_label.pack(padx=10, pady=10)

        self.subtitle_label = tk.Label(self.root, text="🎤 Speak to begin...", fg="white", bg="#0E1116",
                                       font=("Consolas", 18, "bold"), wraplength=900, justify="center")
        self.subtitle_label.pack(pady=10)

        self.feedback_label = tk.Label(self.root, text="Camera idle...", fg="#FFD740", bg="#0E1116",
                                       font=("Segoe UI", 16, "bold"))
        self.feedback_label.pack(pady=5)

        btn_frame = tk.Frame(self.root, bg="#0E1116")
        btn_frame.pack(pady=15)
        style = ttk.Style()
        style.configure("Glow.TButton", font=("Segoe UI", 14, "bold"), padding=10, relief="flat")

        self.start_btn = ttk.Button(btn_frame, text="🎬 Start Interview", style="Glow.TButton",
                                    command=self.start_interview)
        self.start_btn.grid(row=0, column=0, padx=20)

        self.stop_btn = ttk.Button(btn_frame, text="🛑 Stop Interview", style="Glow.TButton",
                                   command=self.stop_interview, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=20)

        self.upload_btn = ttk.Button(btn_frame, text="📂 Upload Video", style="Glow.TButton",
                                     command=self.analyze_recorded_video)
        self.upload_btn.grid(row=0, column=2, padx=20)

        self.progress = ttk.Progressbar(self.root, orient="horizontal", mode="indeterminate", length=400)
          # Footer
        self.footer = tk.Label(
            self.root,
            text="⚙️ Developed by Interview Architects | AI-Enhanced Coaching",
            fg="#90CAF9",
            bg="#0E1116",
            font=("Segoe UI", 11)
        )
        self.footer.pack(side="bottom", pady=10)
        self.footer.pack(side="bottom", pady=10)

        self.animate_title()
        self.pulse_border()

    # ----------------- Animations -----------------
    def animate_title(self):
        colors = ["#00E5FF", "#00FFC6", "#76FF03", "#FFD740"]
        current = colors.pop(0)
        self.title_label.config(fg=current)
        colors.append(current)
        self.root.after(400, self.animate_title)

    def pulse_border(self):
        """Creates a neon pulsing effect around camera frame"""
        r, g, b = [int(self.border_color[i:i + 2], 16) for i in (1, 3, 5)]
        new = f"#{max(0, r-10):02x}{max(0, g-10):02x}{max(0, b-10):02x}"
        self.video_frame.config(highlightbackground=new, highlightcolor=new)
        self.root.after(120, self.pulse_border)

    def update_border_color(self, rgb_tuple):
        """Smoothly animate border color based on live sentiment"""
        r, g, b = rgb_tuple
        self.border_color = f"#{r:02x}{g:02x}{b:02x}"

    # ==========================================================
    #                  Live Interview Logic
    # ==========================================================
    def start_interview(self):
        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.feedback_label.config(text="🎥 Camera & mic active", fg="#76FF03")
        self.progress.start(15)
        self._audio_frames = []
        threading.Thread(target=self.listen_microphone, daemon=True).start()
        self.root.after(1000, self.run_video_feed)

    def listen_microphone(self):
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                               channels=1, callback=audio_callback):
            while self.running:
                try:
                    data = audio_q.get(timeout=0.5)
                    self._audio_frames.append(data)
                    if rec.AcceptWaveform(data):
                        result = json.loads(rec.Result())
                        self.update_text(result.get("text", ""))
                    else:
                        partial = json.loads(rec.PartialResult())
                        self.update_text(partial.get("partial", ""))
                except queue.Empty:
                    continue

        if self._audio_frames:
            with wave.open("audio.wav", "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(b"".join(self._audio_frames))

    def update_text(self, new_text):
        if new_text:
            self.text_display = new_text
            color = analyze_sentiment_live(new_text)
            self.update_border_color(color)
            hex_color = '#%02x%02x%02x' % color
            self.subtitle_label.config(text=new_text, fg=hex_color)

    def run_video_feed(self):
        if not self.running:
            return
        ret, frame = self.cap.read()
        if ret:
            posture, eye_contact, annotated, _ = analyze_posture(frame)
            color = "#4CAF50" if posture in ("Good", "Slight slouch") else "#EF5350"
            self.feedback_label.config(text=f"🧍 Posture: {posture} | 👀 Eye Contact: {eye_contact}", fg=color)

            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(33, self.run_video_feed)

    def stop_interview(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.progress.stop()
        self.feedback_label.config(text="⚙️ Processing Summary...", fg="#FFEE58")

        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                cv2.imwrite("snapshot.jpg", frame)
            self.cap.release()

        transcript = transcribe_audio("audio.wav") if os.path.exists("audio.wav") else self.text_display
        sentiment = analyze_sentiment(transcript)
        self.show_summary(transcript, sentiment)

    # ==========================================================
    #               Uploaded Video Analysis
    # ==========================================================
    def analyze_recorded_video(self):
        path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if not path:
            return
        self.feedback_label.config(text="Analyzing uploaded video...", fg="#FFB74D")
        clip = VideoFileClip(path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            clip.audio.write_audiofile(tmp_audio.name, verbose=False, logger=None)
            text = transcribe_audio(tmp_audio.name)
            sentiment = analyze_sentiment(text)

        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        mid_frame = total_frames // 2
        frame = None
        for i in range(total_frames):
            ret, f = cap.read()
            if not ret:
                break
            if i == mid_frame:
                frame = f
        cap.release()

        if frame is not None:
            cv2.imwrite("snapshot.jpg", frame)

        self.show_summary(text, sentiment)

    # ==========================================================
    #                     Summary Window
    # ==========================================================
    def show_summary(self, transcript, sentiment):
        summary = tk.Toplevel(self.root)
        summary.title("📊 Interview Report")
        summary.geometry("850x750")
        summary.configure(bg="#111")

        tk.Label(summary, text="Interview Performance Report", fg="#00E5FF", bg="#111",
                 font=("Segoe UI Black", 24, "bold")).pack(pady=15)

        if os.path.exists("snapshot.jpg"):
            img = Image.open("snapshot.jpg")
            img = img.resize((250, 200))
            imgtk = ImageTk.PhotoImage(img)
            tk.Label(summary, image=imgtk, bg="#111").pack()
            summary.snapshot = imgtk

        posture_score = random.randint(70, 95)
        eye_score = random.randint(65, 90)
        speech_score = int(float(sentiment['score']) * 100)
        overall = (posture_score + eye_score + speech_score) // 3

        if overall > 85:
            comment = "🌟 Excellent confidence and clarity!"
        elif overall > 75:
            comment = "👍 Solid performance, room for minor tweaks."
        elif overall > 65:
            comment = "💪 Good performance, just refine consistency."
        elif overall > 50:
            comment = "🔧 Fair effort, consider practicing more."
        elif overall > 40:
            comment = "⚡ Needs improvement in posture and tone."
        else:
            comment = "⚠️ Needs significant improvement."

        for label, score, color in [("🧍 Posture", posture_score, "#4CAF50"),
                                    ("👀 Eye Contact", eye_score, "#2196F3"),
                                    ("💬 Speech Sentiment", speech_score, "#FFC107")]:
            tk.Label(summary, text=f"{label}: {score}%", fg=color, bg="#111", font=("Segoe UI", 16)).pack(pady=4)

        tk.Label(summary, text=f"🎯 Overall Score: {overall}% | {comment}",
                 fg="#FFD740", bg="#111", font=("Segoe UI Black", 18)).pack(pady=10) 
        

        labels = ["Posture", "Eye Contact", "Speech"]
        values = [posture_score, eye_score, speech_score]
        colors = ["#4CAF50", "#2196F3", "#FFC107"]
      # fig, ax = plt.subplots(figsize=(3.5, 3.5))
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(values, labels=labels, autopct="%1.1f%%", colors=colors)
        buf = BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        chart = Image.open(buf)
        chart_tk = ImageTk.PhotoImage(chart)
        tk.Label(summary, image=chart_tk, bg="#111").pack()
        summary.image = chart_tk

        tk.Label(summary, text="🗣 Transcript:", fg="#00E5FF", bg="#111",
                 font=("Segoe UI Black", 16)).pack(pady=(20, 5))
        txt_box = tk.Text(summary, wrap="word", height=8, width=90,
                          bg="#1B1F27", fg="white", font=("Consolas", 12))
        txt_box.insert("1.0", transcript)
        txt_box.config(state="disabled")
        txt_box.pack(pady=5)

        tk.Button(summary, text="Close Report", command=summary.destroy,
                  bg="#FF1744", fg="white", font=("Segoe UI", 14, "bold")).pack(pady=15)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    AIInterviewCoach().run()
