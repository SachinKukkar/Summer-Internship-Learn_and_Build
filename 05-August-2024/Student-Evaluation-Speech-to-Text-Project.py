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


class NLPQuizApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('NLP Quiz')
        self.geometry('500x400')  # Adjusted size to fit the new button
        self.configure(bg='#f0f0f0')

        # Initialize NLP components
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

        # Load questions and answers
        self.questions_answers = self.load_questions_answers('questions_answers.csv')
        self.current_question_index = 0
        self.score = 0
        self.total_questions = len(self.questions_answers)

        # Create and place widgets
        self.create_widgets()

    def create_widgets(self):
        self.question_label = tk.Label(self, text='', bg='#f0f0f0', font=('Arial', 16), wraplength=400)
        self.question_label.pack(pady=20)

        self.answer_entry = tk.Entry(self, width=50, font=('Arial', 14))
        self.answer_entry.pack(pady=10)

        self.submit_button = tk.Button(self, text='Submit Answer', command=self.submit_answer, bg='#4CAF50', fg='white', font=('Arial', 14))
        self.submit_button.pack(pady=10)

        self.result_label = tk.Label(self, text='', bg='#f0f0f0', font=('Arial', 14))
        self.result_label.pack(pady=10)

        self.start_quiz_button = tk.Button(self, text='Start Quiz', command=self.start_quiz, bg='#4CAF50', fg='white', font=('Arial', 14))
        self.start_quiz_button.pack(pady=10)

        self.progress_label = tk.Label(self, text='', bg='#f0f0f0', font=('Arial', 12))
        self.progress_label.pack(pady=10)

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
        # Tokenize, remove stopwords, and lemmatize the text
        tokens = nltk.word_tokenize(text.lower())
        filtered_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(filtered_tokens)

    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()

    def submit_answer(self):
        user_answer = self.answer_entry.get().strip()
        correct_answer = self.questions_answers[self.current_question_index][1].strip()

        preprocessed_user_answer = self.preprocess_text(user_answer)
        preprocessed_correct_answer = self.preprocess_text(correct_answer)

        user_embedding = self.get_embedding(preprocessed_user_answer)
        correct_embedding = self.get_embedding(preprocessed_correct_answer)

        similarity_score = cosine_similarity(user_embedding, correct_embedding)[0][0]

        print(f"Similarity Score: {similarity_score:.4f}")

        threshold = 0.5  
        if similarity_score > threshold:
            self.score += 1
            result_text = (f"Correct!\n\n"
                           f"Entered Answer: '{user_answer}'\n"
                           f"Correct Answer: '{correct_answer}'\n"
                           f"Similarity Score: {similarity_score * 100:.2f}%")
        else:
            result_text = (f"Incorrect.\n\n"
                           f"Entered Answer: '{user_answer}'\n"
                           f"Correct Answer: '{correct_answer}'\n"
                           f"Similarity Score: {similarity_score * 100:.2f}%")

        self.result_label.config(text=result_text)

        # next question
        self.current_question_index += 1
        self.display_next_question()

    def show_score_analysis(self):
        result_text = f"Quiz completed!\n\nYour Score: {self.score}/{self.total_questions}"
        percentage = (self.score / self.total_questions) * 100
        result_text += f"\nYour Percentage: {percentage:.2f}%"
        messagebox.showinfo("Quiz Result", result_text)
        self.result_label.config(text=result_text)

    def update_progress(self):
        progress_text = f"Question {self.current_question_index + 1}/{self.total_questions}"
        self.progress_label.config(text=progress_text)


if __name__ == "__main__":
    app = NLPQuizApp()
    app.mainloop()
