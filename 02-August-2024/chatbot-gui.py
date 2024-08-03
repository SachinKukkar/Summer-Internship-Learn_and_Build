import requests
import tkinter as tk
from tkinter import scrolledtext

# API URL and headers
API_URL = "https://api-inference.huggingface.co/models/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def get_response():
    question = question_entry.get("1.0", "end").strip()
    context = context_entry.get("1.0", "end").strip()

    if not question or not context:
        response_entry.config(state="normal")
        response_entry.delete("1.0", "end")
        response_entry.insert("end", "Both question and context must be provided.")
        response_entry.config(state='disabled')
        return

    payload = {
        "inputs": {
            "question": question,
            "context": context
        }
    }

    try:
        output = query(payload)
        answer = output.get('answer', 'No answer found in response.')
    except Exception as e:
        answer = f"An error occurred: {e}"

    response_entry.config(state="normal")
    response_entry.delete("1.0", "end")
    response_entry.insert('end', answer)
    response_entry.config(state='disabled')    

# Create the main window
root = tk.Tk()
root.title("GenAI Q&A App")
root.geometry("900x700")
root.configure(bg='#f4f4f4')

# Title label
title_label = tk.Label(root, text="GenAI Question & Answer App", font=('Helvetica', 24, 'bold'), bg='#f4f4f4', fg='#333333')
title_label.pack(pady=20)

# Frame for question input
input_frame = tk.Frame(root, bg='#ffffff', bd=2, relief='solid')
input_frame.pack(padx=20, pady=10, fill='x')

# Question input
question_label = tk.Label(input_frame, text='Enter your question:', font=('Helvetica', 16), bg='#ffffff', fg='#333333')
question_label.pack(anchor='w', padx=10, pady=5)

question_entry = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=110, height=4, font=('Helvetica', 14), bg='#ffffff', fg='#000000')
question_entry.pack(padx=10, pady=5)

# Context input
context_label = tk.Label(input_frame, text='Enter context:', font=('Helvetica', 16), bg='#ffffff', fg='#333333')
context_label.pack(anchor='w', padx=10, pady=5)

context_entry = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, width=110, height=6, font=('Helvetica', 14), bg='#ffffff', fg='#000000')
context_entry.pack(padx=10, pady=5)

# Get Response Button
get_response_button = tk.Button(root, text='Get Response', font=('Helvetica', 16), command=get_response, bg='#00796b', fg='white', padx=20, pady=10, relief='flat')
get_response_button.pack(pady=20)

# Frame for the response
response_frame = tk.Frame(root, bg='#ffffff', bd=2, relief='solid')
response_frame.pack(padx=20, pady=10, fill='both', expand=True)

response_label = tk.Label(response_frame, text='Model Response:', font=('Helvetica', 16), bg='#ffffff', fg='#333333')
response_label.pack(anchor='w', padx=10, pady=5)

response_entry = scrolledtext.ScrolledText(response_frame, wrap=tk.WORD, width=110, height=10, font=('Helvetica', 14), bg='#ffffff', fg='#000000', state='disabled')
response_entry.pack(padx=10, pady=5, fill='both', expand=True)

# Start the GUI event loop
root.mainloop()
