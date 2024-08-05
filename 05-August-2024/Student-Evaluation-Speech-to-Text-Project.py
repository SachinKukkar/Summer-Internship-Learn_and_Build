import tkinter as tk
import tkinter.messagebox as messagebox
import speech_recognition as sr
import threading
import pyaudio 
import wave
import os
import numpy as np

class SpeechToTextApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Speech to Text Converter')
        self.geometry('400x300')
        self.configure(bg='#f0f0f0')  # Set background color of the window

        # Create a folder to save recordings
        self.recordings_folder = 'recordings'
        if not os.path.exists(self.recordings_folder):
            os.makedirs(self.recordings_folder)
        
        self.audio_file_path = os.path.join(self.recordings_folder, 'recorded_audio.wav')
        self.recording = False

        # Create and place widgets with improved colors and styles
        self.record_button = tk.Button(self, text='Start Recording', command=self.start_recording, bg='#4CAF50', fg='white', font=('Arial', 12))
        self.record_button.pack(pady=10)

        self.stop_record_button = tk.Button(self, text='Stop Recording', command=self.stop_recording, state=tk.DISABLED, bg='#f44336', fg='white', font=('Arial', 12))
        self.stop_record_button.pack(pady=10)

        self.play_record_button = tk.Button(self, text='Play Recording', command=self.play_recording, state=tk.DISABLED, bg='#2196F3', fg='white', font=('Arial', 12))
        self.play_record_button.pack(pady=10)

        self.convert_record_button = tk.Button(self, text='Convert Recording to Text', command=self.convert_recording_to_text, state=tk.DISABLED, bg='#FFC107', fg='black', font=('Arial', 12))
        self.convert_record_button.pack(pady=10)

        # Audio level indicator
        self.audio_level_canvas = tk.Canvas(self, width=300, height=30, bg='#e0e0e0', borderwidth=2, relief='sunken')
        self.audio_level_canvas.pack(pady=20)

    def start_recording(self):
        self.recording = True
        self.record_button.config(state=tk.DISABLED)
        self.stop_record_button.config(state=tk.NORMAL)
        self.play_record_button.config(state=tk.DISABLED)
        self.convert_record_button.config(state=tk.DISABLED)

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
        self.frames = []
        self.recording_thread = threading.Thread(target=self.record)
        self.recording_thread.start()

    def record(self):
        while self.recording:
            data = self.stream.read(1024)
            self.frames.append(data)
            # Calculate audio level
            audio_data = np.frombuffer(data, dtype=np.int16)
            audio_level = np.abs(audio_data).mean()
            self.update_audio_level(audio_level)

    def update_audio_level(self, level):
        max_level = 1000  # Adjust this value based on expected input range
        normalized_level = min(level / max_level, 1.0)
        self.audio_level_canvas.delete('all')
        self.audio_level_canvas.create_rectangle(0, 0, 300 * normalized_level, 30, fill='green')

    def stop_recording(self):
        self.recording = False
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()  # Properly terminate the PyAudio instance

        with wave.open(self.audio_file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b''.join(self.frames))

        self.record_button.config(state=tk.NORMAL)
        self.stop_record_button.config(state=tk.DISABLED)
        self.play_record_button.config(state=tk.NORMAL)
        self.convert_record_button.config(state=tk.NORMAL)

    def play_recording(self):
        if os.name == 'nt':  # Windows
            os.system(f'start {self.audio_file_path}')
        else:  # Unix-based
            os.system(f'aplay {self.audio_file_path}')

    def convert_recording_to_text(self):
        rs = sr.Recognizer()
        try:
            with sr.AudioFile(self.audio_file_path) as source:
                audio_data = rs.record(source)
                text = rs.recognize_google(audio_data)
                messagebox.showinfo("Speech to Text", text)
        except sr.UnknownValueError:
            messagebox.showwarning("Speech to Text", "Could not recognize the audio!")
        except sr.RequestError as e:
            messagebox.showerror("Speech to Text", f"Error occurred: {e}")

if __name__ == "__main__":
    app = SpeechToTextApp()
    app.mainloop()
