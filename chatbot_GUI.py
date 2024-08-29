import tkinter as tk
from tkinter import scrolledtext
import json
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import random


nltk.download('punkt_tab')
nltk.download('wordnet')
print("Preparation Completed")


model = load_model('chatbot_model.h5')

lemmatizer = WordNetLemmatizer()

with open('intents.json') as file:
    intents = json.load(file)

words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words]))
classes = sorted(classes)

# Function to preprocess the input and get the response
def preprocess_input(user_input):
    # Tokenize and lemmatize user input
    tokens = nltk.word_tokenize(user_input)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    
    # Create a bag of words representation
    bag = [0] * len(words)
    for w in tokens:
        if w in words:
            bag[words.index(w)] = 1
            
    return np.array([bag])

# Function to get response from the model
def get_response(user_input):
    bag = preprocess_input(user_input)
    result = model.predict(bag)[0]  # Predict using the loaded Keras model
    threshold = 0.5  # Set a threshold to consider a response
    results = [[i, r] for i, r in enumerate(result) if r > threshold]

    # Sort by probability
    results.sort(key=lambda x: x[1], reverse=True)

    if results:
        tag = classes[results[0][0]]
        for intent in intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])  # Return a random response from the matched intent

    return "Sorry, I didn't understand that. Can you try again?"

# Create the main application window
root = tk.Tk()
root.title("Chatbot Interface")
root.geometry("500x600")

# Chat display area
chat_display = scrolledtext.ScrolledText(root, wrap=tk.WORD, state='disabled', width=60, height=25)
chat_display.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# User input field
user_input = tk.Entry(root, width=50)
user_input.grid(row=1, column=0, padx=10, pady=10)

# Function to handle user input and display chatbot response
def send_message():
    message = user_input.get()
    if message:
        chat_display.config(state='normal')
        chat_display.insert(tk.END, f"You: {message}\n")
        user_input.delete(0, tk.END)
        
        # Get the response from the chatbot
        response = get_response(message)
        chat_display.insert(tk.END, f"Bot: {response}\n")
        chat_display.config(state='disabled')
        chat_display.yview(tk.END)

# Send button to trigger the response
send_button = tk.Button(root, text="Send", command=send_message)
send_button.grid(row=1, column=1, padx=10, pady=10)

# Run the main loop
root.mainloop()