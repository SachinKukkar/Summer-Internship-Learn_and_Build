import tkinter as tk
import tkinter.messagebox as messagebox
import speech_recognition as sr
import threading
import pyaudio
import wave
import os
import numpy as np
import spacy
import random

class SpeechToTextApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Speech to Text Converter')
        self.geometry('600x600')  # Increased size for better layout
        self.configure(bg='#f0f0f0')

        # Load the NLP model
        self.nlp = spacy.load("en_core_web_sm")

        # Create a folder to save recordings
        self.recordings_folder = 'recordings'
        if not os.path.exists(self.recordings_folder):
            os.makedirs(self.recordings_folder)

        self.audio_file_path = os.path.join(self.recordings_folder, 'recorded_audio.wav')
        self.recording = False

        # Initialize question index and score
        self.current_question_index = 0
        self.total_questions = 0
        self.score = 0

        # Python-related questions and answers
        self.questions = [
            ("What is the keyword used to create a function in Python?", "def"),
            ("How do you insert comments in Python code?", "hash symbol (#)"),
            ("What data type is used to store a sequence of characters?", "string"),
            ("Which function is used to read user input in Python?", "input"),
            ("How do you create a list in Python?", "list() or using square brackets []"),
            ("What is the result of 5 // 2 in Python?", "2"),
            ("Which method is used to remove whitespace from the beginning and end of a string?", "strip"),
            ("How do you create a class in Python?", "class Keyword"),
            ("What does the 'self' keyword represent in a class?", "Instance of the class"),
            ("How do you handle exceptions in Python?", "try...except"),
            # Add more questions as needed
        ]

        # Create and place widgets
        self.question_frame = tk.Frame(self, bg='#f0f0f0')
        self.question_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.question_text = tk.Text(self.question_frame, height=4, width=70, wrap=tk.WORD, font=('Arial', 14))
        self.question_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scrollbar = tk.Scrollbar(self.question_frame, command=self.question_text.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.question_text.config(yscrollcommand=self.scrollbar.set)

        self.num_questions_label = tk.Label(self, text='Select number of questions:', bg='#f0f0f0', font=('Arial', 12))
        self.num_questions_label.pack(pady=10)

        self.num_questions_var = tk.IntVar(value=1)
        self.num_questions_spinbox = tk.Spinbox(self, from_=1, to=len(self.questions), textvariable=self.num_questions_var, font=('Arial', 12))
        self.num_questions_spinbox.pack(pady=10)

        self.start_quiz_button = tk.Button(self, text='Start Quiz', command=self.start_quiz, bg='#4CAF50', fg='white', font=('Arial', 12))
        self.start_quiz_button.pack(pady=10)

        self.record_button = tk.Button(self, text='Start Recording', command=self.start_recording, bg='#4CAF50', fg='white', font=('Arial', 12), state=tk.DISABLED)
        self.record_button.pack(pady=10)

        self.stop_record_button = tk.Button(self, text='Stop Recording', command=self.stop_recording, state=tk.DISABLED, bg='#f44336', fg='white', font=('Arial', 12))
        self.stop_record_button.pack(pady=10)

        self.play_record_button = tk.Button(self, text='Play Recording', command=self.play_recording, state=tk.DISABLED, bg='#2196F3', fg='white', font=('Arial', 12))
        self.play_record_button.pack(pady=10)

        self.convert_record_button = tk.Button(self, text='Evaluate Answer', command=self.convert_recording_to_text, state=tk.DISABLED, bg='#FFC107', fg='black', font=('Arial', 12))
        self.convert_record_button.pack(pady=10)

        self.result_label = tk.Label(self, text='', bg='#f0f0f0', font=('Arial', 12))
        self.result_label.pack(pady=10)

        # Audio level indicator
        self.audio_level_canvas = tk.Canvas(self, width=300, height=30, bg='#e0e0e0', borderwidth=2, relief='sunken')
        self.audio_level_canvas.pack(pady=20)

    def start_quiz(self):
        self.current_question_index = 0
        self.score = 0
        self.total_questions = self.num_questions_var.get()

        if self.total_questions > len(self.questions):
            messagebox.showwarning("Warning", "Number of questions exceeds available questions.")
            return

        self.selected_questions = random.sample(self.questions, self.total_questions)
        self.display_next_question()

    def display_next_question(self):
        if self.current_question_index < self.total_questions:
            question = self.selected_questions[self.current_question_index][0]
            self.question_text.delete(1.0, tk.END)
            self.question_text.insert(tk.END, question)
            self.record_button.config(state=tk.NORMAL)
            self.stop_record_button.config(state=tk.DISABLED)
            self.play_record_button.config(state=tk.DISABLED)
            self.convert_record_button.config(state=tk.DISABLED)
        else:
            self.show_score_analysis()

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
        self.audio.terminate()

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
                self.evaluate_answer(text)
        except sr.UnknownValueError:
            messagebox.showwarning("Speech to Text", "Could not recognize the audio!")
        except sr.RequestError as e:
            messagebox.showerror("Speech to Text", f"Error occurred: {e}")

    def evaluate_answer(self, answer_text):
        question = self.selected_questions[self.current_question_index][0]
        correct_answer = self.selected_questions[self.current_question_index][1]

        # Simple text matching
        if correct_answer.lower() in answer_text.lower():
            self.score += 1
            result_text = "Correct!"
        else:
            result_text = "Incorrect."

        self.result_label.config(text=f"Question: {question}\nYour Answer: {answer_text}\n{result_text}")
        messagebox.showinfo("Evaluation Result", f"{result_text}\nCorrect Answer: {correct_answer}")

        # Move to the next question
        self.current_question_index += 1
        self.display_next_question()

    def show_score_analysis(self):
        result_text = f"Quiz completed!\n\nYour Score: {self.score}/{self.total_questions}"
        percentage = (self.score / self.total_questions) * 100
        result_text += f"\nYour Percentage: {percentage:.2f}%"

        self.result_label.config(text=result_text)
        messagebox.showinfo("Quiz Completed", result_text)

if __name__ == "__main__":
    app = SpeechToTextApp()
    app.mainloop()
