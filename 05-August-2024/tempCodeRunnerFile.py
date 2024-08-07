import tkinter as tk
import tkinter.messagebox as messagebox
import nltk
import csv
import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from pydub import AudioSegment
import sounddevice as sd
import soundfile as sf
import os
import speech_recognition as sr

class NLPQuizApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('NLP Quiz')
        self.geometry('600x700')
        self.configure(bg='#f8f9fa')

        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')

        # Load questions and answers
        self.questions_answers = self.load_questions_answers('questions_answers.csv')
        self.current_question_index = 0
        self.score = 0
        self.total_questions = len(self.questions_answers)

        # Variables for audio recording
        self.recording = False
        self.audio_file = 'recording.wav'
        self.audio_data = []

        # Create and place widgets
        self.create_widgets()

    def create_widgets(self):
        self.question_label = tk.Label(self, text='', bg='#f8f9fa', font=('Helvetica', 16, 'bold'), wraplength=500)
        self.question_label.pack(pady=20)

        self.answer_entry = tk.Entry(self, width=50, font=('Helvetica', 14))
        self.answer_entry.pack(pady=10)

        self.start_quiz_button = tk.Button(self, text='Start Quiz', command=self.start_quiz, bg='#007bff', fg='white', font=('Helvetica', 14, 'bold'), relief=tk.RAISED, padx=10, pady=5)
        self.start_quiz_button.pack(pady=10)

        self.record_button = tk.Button(self, text='Start Recording', command=self.start_recording, bg='#28a745', fg='white', font=('Helvetica', 14, 'bold'), relief=tk.RAISED, padx=10, pady=5)
        self.record_button.pack(pady=10)

        self.stop_record_button = tk.Button(self, text='Stop Recording', command=self.stop_recording, bg='#dc3545', fg='white', font=('Helvetica', 14, 'bold'), relief=tk.RAISED, padx=10, pady=5, state=tk.DISABLED)
        self.stop_record_button.pack(pady=10)

        self.convert_audio_button = tk.Button(self, text='Convert Audio to Text', command=self.convert_audio_to_text, bg='#17a2b8', fg='white', font=('Helvetica', 14, 'bold'), relief=tk.RAISED, padx=10, pady=5, state=tk.DISABLED)
        self.convert_audio_button.pack(pady=10)

        self.submit_answer_button = tk.Button(self, text='Submit Answer', command=self.submit_answer, bg='#ffc107', fg='black', font=('Helvetica', 14, 'bold'), relief=tk.RAISED, padx=10, pady=5, state=tk.DISABLED)
        self.submit_answer_button.pack(pady=10)

        self.converted_text_label = tk.Label(self, text='', bg='#f8f9fa', font=('Helvetica', 12))
        self.converted_text_label.pack(pady=10)

        self.result_label = tk.Label(self, text='', bg='#f8f9fa', font=('Helvetica', 14, 'bold'))
        self.result_label.pack(pady=10)

        self.progress_label = tk.Label(self, text='', bg='#f8f9fa', font=('Helvetica', 12))
        self.progress_label.pack(pady=10)

        self.similarity_label = tk.Label(self, text='', bg='#f8f9fa', font=('Helvetica', 12))
        self.similarity_label.pack(pady=10)

        self.user_answer_label = tk.Label(self, text='', bg='#f8f9fa', font=('Helvetica', 12))
        self.user_answer_label.pack(pady=10)

        self.correct_answer_label = tk.Label(self, text='', bg='#f8f9fa', font=('Helvetica', 12))
        self.correct_answer_label.pack(pady=10)

    def load_questions_answers(self, file_path):
        questions_answers = []
        try:
            with open(file_path, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                next(reader)  # Skip the header row
                for row in reader:
                    if len(row) == 2:
                        question, answer = row
                        questions_answers.append((question, answer))
                    else:
                        print(f"Skipping invalid row: {row}")
        except FileNotFoundError:
            messagebox.showerror("Error", "Questions and Answers file not found.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
        return questions_answers

    def start_quiz(self):
        self.current_question_index = 0
        self.score = 0
        self.display_next_question()

    def display_next_question(self):
        if self.current_question_index < self.total_questions:
            question, _ = self.questions_answers[self.current_question_index]
            self.question_label.config(text=question)
            self.answer_entry.delete(0, tk.END)
            self.update_progress()
        else:
            self.show_score_analysis()

    def preprocess_text(self, text):
        tokens = nltk.word_tokenize(text.lower())
        filtered_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(filtered_tokens)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def submit_answer(self):
        self.compare_answer(self.converted_text_label.cget("text"))

    def show_score_analysis(self):
        result_text = f"Quiz completed!\n\nYour Score: {self.score}/{self.total_questions}"
        percentage = (self.score / self.total_questions) * 100
        result_text += f"\nYour Percentage: {percentage:.2f}%"
        messagebox.showinfo("Quiz Result", result_text)
        self.result_label.config(text=result_text)

    def update_progress(self):
        progress_text = f"Question {self.current_question_index + 1}/{self.total_questions}"
        self.progress_label.config(text=progress_text)

    def start_recording(self):
        self.recording = True
        self.record_button.config(state=tk.DISABLED)
        self.stop_record_button.config(state=tk.NORMAL)
        self.convert_audio_button.config(state=tk.DISABLED)
        self.submit_answer_button.config(state=tk.DISABLED)
        self.audio_data = []

        def callback(indata, frames, time, status):
            if status:
                print(status)
            self.audio_data.append(indata.copy())

        self.stream = sd.InputStream(callback=callback)
        self.stream.start()

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.stream.stop()
            self.stream.close()

            # Save the recorded data as a WAV file
            self.audio_data = np.concatenate(self.audio_data, axis=0)
            sf.write(self.audio_file, self.audio_data, 44100)

            self.stop_record_button.config(state=tk.DISABLED)
            self.convert_audio_button.config(state=tk.NORMAL)

    def convert_audio_to_text(self):
        if os.path.exists(self.audio_file):
            recognizer = sr.Recognizer()
            with sr.AudioFile(self.audio_file) as source:
                audio_data = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio_data)
                    self.converted_text_label.config(text=f"Converted Text: {text}")
                    self.submit_answer_button.config(state=tk.NORMAL)
                except sr.UnknownValueError:
                    messagebox.showerror("Error", "Google Speech Recognition could not understand audio")
                except sr.RequestError as e:
                    messagebox.showerror("Error", f"Could not request results from Google Speech Recognition service; {e}")
        else:
            messagebox.showerror("Error", "No audio file found. Please record first.")

    def compare_answer(self, user_answer_text):
        _, correct_answer = self.questions_answers[self.current_question_index]
        user_answer_text = self.preprocess_text(user_answer_text)
        correct_answer = self.preprocess_text(correct_answer)

        user_embedding = self.get_embedding(user_answer_text)
        correct_embedding = self.get_embedding(correct_answer)

        similarity = cosine_similarity(user_embedding, correct_embedding)[0][0]
        self.similarity_label.config(text=f"Similarity Score: {similarity:.2f}")

        if similarity > 0.8:  # Adjust threshold as needed
            self.score += 1
            self.result_label.config(text="Correct!")
        else:
            self.result_label.config(text="Incorrect.")
        
        self.current_question_index += 1
        self.display_next_question()

        # Re-enable the Start Recording button
        self.record_button.config(state=tk.NORMAL)

if __name__ == "__main__":
    app = NLPQuizApp()
    app.mainloop()
