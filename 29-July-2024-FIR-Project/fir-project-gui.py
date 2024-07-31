import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer, util
from tkinter import Tk, Label, Entry, Text, Button, Scrollbar, RIGHT, Y, END, Frame
from tkinter import ttk
from threading import Thread
import time

# Preprocess function
def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    preprocessed_text = ' '.join(words)
    return preprocessed_text

# Load preprocessed data and model
new_ds = pickle.load(open('preprocess_data.pkl', 'rb'))
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Suggest sections function
def suggest_sections(complaint, dataset, min_suggestions=5):
    preprocessed_complaint = preprocess_text(complaint)
    complaint_embedding = model.encode(preprocessed_complaint)
    section_embedding = model.encode(dataset['Combo'].tolist())
    similarities = util.pytorch_cos_sim(complaint_embedding, section_embedding)[0]
    similarity_threshold = 0.2
    relevant_indices = []
    while len(relevant_indices) < min_suggestions and similarity_threshold > 0:
        relevant_indices = [i for i, sim in enumerate(similarities) if sim > similarity_threshold]
        similarity_threshold -= 0.05
    sorted_indices = sorted(relevant_indices, key=lambda i: similarities[i], reverse=True)
    suggestions = dataset.iloc[sorted_indices][['Description', 'Offense', 'Punishment', 'Cognizable', 'Bailable', 'Court', 'Combo']].to_dict(orient='records')
    return suggestions

# Function to run the suggestion generation in a separate thread
def process_suggestions():
    complaint = complaint_entry.get()
    show_loading_animation()
    # Simulate a delay for demonstration purposes
    time.sleep(2)
    suggestions = suggest_sections(complaint, new_ds)
    hide_loading_animation()
    update_output_text(suggestions)

# Function to show loading animation
def show_loading_animation():
    loading_label.pack()
    progress_bar.pack()
    progress_bar.start()
    root.update_idletasks()  # Update the GUI

# Function to hide loading animation
def hide_loading_animation():
    loading_label.pack_forget()
    progress_bar.pack_forget()
    progress_bar.stop()

# Function to update the output text
def update_output_text(suggestions):
    output_text.delete("1.0", END)
    if suggestions:
        output_text.insert(END, "Suggested Sections are:\n\n")
        for suggestion in suggestions:
            output_text.insert(END, f"Description: {suggestion['Description']}\n")
            output_text.insert(END, f"Offense: {suggestion['Offense']}\n")
            output_text.insert(END, f"Punishment: {suggestion['Punishment']}\n")
            output_text.insert(END, "_________________________________________________________________________________________\n\n")
    else:
        output_text.insert(END, "No record is found")

# Function to handle the button click
def on_suggest_button_click():
    # Run the suggestion generation in a separate thread to avoid blocking the GUI
    Thread(target=process_suggestions).start()

root = Tk()
root.title("IPC Section Suggestions")

# Create a main frame with background color
main_frame = Frame(root, padx=20, pady=20, bg='#f0f8ff')  # Light AliceBlue background
main_frame.pack(expand=True, fill='both')

# Header Label
header_label = Label(main_frame, text="IPC Section Suggestions", font=('Helvetica', 18, 'bold'), bg='#f0f8ff', fg='#4682b4')  # SteelBlue color
header_label.pack(pady=10)

# Crime Description Entry
complaint_label = Label(main_frame, text='Enter crime description:', font=('Helvetica', 14), bg='#f0f8ff', fg='#4682b4')
complaint_label.pack(pady=5)

complaint_entry = Entry(main_frame, width=80, font=('Helvetica', 14), bd=2, relief='solid')
complaint_entry.pack(pady=5)

# Suggest Button
suggest_button = Button(main_frame, text='Get Suggestions', command=on_suggest_button_click, font=('Helvetica', 14, 'bold'), bg='#20b2aa', fg='white', bd=0, relief='flat')
suggest_button.pack(pady=10)

# Loading Animation Label
loading_label = Label(main_frame, text="Loading...", font=('Helvetica', 14, 'italic'), bg='#f0f8ff', fg='#ff4500')  # OrangeRed color

# Progress Bar
progress_bar = ttk.Progressbar(main_frame, mode='indeterminate')

# Output Text with Scrollbar
output_frame = Frame(main_frame)
output_frame.pack(expand=True, fill='both')

scrollbar = Scrollbar(output_frame)
scrollbar.pack(side=RIGHT, fill=Y)

output_text = Text(output_frame, width=80, height=20, wrap='word', yscrollcommand=scrollbar.set, font=('Helvetica', 12), bd=2, relief='solid', bg='#f5f5f5', fg='#333333')
output_text.pack(expand=True, fill='both')

scrollbar.config(command=output_text.yview)

root.mainloop()
