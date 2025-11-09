# import sounddevice as sd
# import soundfile as sf
# import wave
# import json
# import numpy as np
# import queue
# import threading
# import scipy.signal
# import time
# from vosk import Model, KaldiRecognizer


# # Loading the Vosk model once globally
# model = Model(model_name="vosk-model-small-en-us-0.15")

# # Global flags / buffers
# q = queue.Queue()
# is_recording = False
# recording_thread = None

# # ---------------------- Audio Callback ----------------------

# def audio_callback(indata, frames, time_, status):
#     """Receive mic audio chunks and push to queue."""
#     if status:
#         print("Audio status:", status)
#     q.put(indata.copy())  # keep float32 copy

# # ---------------------- Start Recording ----------------------

# def start_recording(filename="audio.wav", device=1):
#     """Start continuous audio capture in a background thread."""
#     global is_recording, recording_thread
#     is_recording = True
#     samplerate = 16000
#     print(f"🎙️ Continuous recording started on device {device}...")

#     def _record_worker():
#         # open file once, write continuously
#         with sf.SoundFile(filename, mode='w', samplerate=samplerate,
#                           channels=1, subtype='PCM_16') as file:
#             with sd.InputStream(samplerate=samplerate, channels=1,
#                                 dtype='float32', callback=audio_callback,
#                                 device=device):
#                 while is_recording:
#                     data = q.get()
#                     data_int16 = np.int16(data * 32767)
#                     file.write(data_int16)
#         print("✅ Audio file closed cleanly.")

#     recording_thread = threading.Thread(target=_record_worker, daemon=True)
#     recording_thread.start()

# # ---------------------- Stop Recording ----------------------

# def stop_recording():
#     """Stop the continuous recording and wait for the thread to finish."""
#     global is_recording, recording_thread
#     is_recording = False
#     time.sleep(0.5)  # short delay to let last chunk write
#     if recording_thread and recording_thread.is_alive():
#         recording_thread.join(timeout=2)
#     print("🛑 Recording stopped and saved.")

# # ---------------------- Transcription ----------------------

# def transcribe_audio(filename="audio.wav"):
#     """Transcribe recorded audio using Vosk with detailed debug output."""
#     import soundfile as sf, scipy.signal, wave, json
#     print("🔍 Opening file:", filename)
#     data, rate = sf.read(filename)
#     print(f"Audio shape: {data.shape}, Sample rate: {rate}")

#     # Resample if not 16000 Hz
#     if rate != 16000:
#         print(f"Resampling {rate} → 16000 Hz")
#         data = scipy.signal.resample(data, int(len(data) * 16000 / rate))
#         sf.write(filename, data, 16000)

#     wf = wave.open(filename, "rb")
#     rec = KaldiRecognizer(model, wf.getframerate())
#     text = ""
#     while True:
#         data = wf.readframes(4000)
#         if len(data) == 0:
#             break
#         if rec.AcceptWaveform(data):
#             res = json.loads(rec.Result())
#             print("Partial result:", res)
#             text += res.get("text", "") + " "
#     final = json.loads(rec.FinalResult())
#     text += final.get("text", "")
#     print("Final result:", final)
#     print("🗣️ Transcribed text:", text.strip())
#     return text.strip()

# # ---------------------- Standalone Test ----------------------

# if __name__ == "__main__":
#     start_recording("test.wav", device=1)  # 👈 change to 1, 2, or 18 if needed
#     print("🎙 Recording for 6 seconds...")
#     time.sleep(6)
#     stop_recording()
#     print("✅ Recording stopped. Transcribing...")
#     print("Transcript:", transcribe_audio("test.wav"))


# models/speech_model.py
import soundfile as sf
import wave
import json
import scipy.signal
from vosk import Model, KaldiRecognizer
import os

_global_vosk_model = None

def load_vosk_model(path="vosk-model-small-en-us-0.15"):
    """
    Load the Vosk model from `path`. Returns a Model instance.
    Raises FileNotFoundError if not present.
    """
    global _global_vosk_model
    if _global_vosk_model is not None:
        return _global_vosk_model

    if not os.path.exists(path):
        raise FileNotFoundError(f"Vosk model not found at {path}")
    _global_vosk_model = Model(path)
    return _global_vosk_model

def transcribe_audio(filename, model=None):
    """
    Transcribe a WAV file using a Vosk model instance.
    If model is None, tries to load default path.
    Returns the transcribed text (string).
    """
    if model is None:
        model = load_vosk_model()

    # ensure wav format readable
    data, rate = sf.read(filename)
    if rate != 16000:
        # resample to 16000
        num_samples = int(len(data) * 16000 / rate)
        data = scipy.signal.resample(data, num_samples)
        # overwrite file with resampled data
        sf.write(filename, data, 16000)
    wf = wave.open(filename, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())

    text = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            text += res.get("text", "") + " "
    final = json.loads(rec.FinalResult())
    text += final.get("text", "")
    return text.strip()
