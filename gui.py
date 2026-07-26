import tkinter as tk
from tkinter import messagebox
import pickle

# Load model
with open("models/spam_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)


def predict():
    text = entry.get("1.0", tk.END).strip()

    if text == "":
        messagebox.showwarning("Warning", "Please enter a message.")
        return

    vector = vectorizer.transform([text])
    result = model.predict(vector)

    if result[0] == 1:
        output.config(text="Prediction : SPAM", fg="red")
    else:
        output.config(text="Prediction : HAM", fg="green")


root = tk.Tk()
root.title("Spam Email Classifier")
root.geometry("600x400")

title = tk.Label(
    root,
    text="Spam Email Classifier",
    font=("Arial", 20, "bold")
)
title.pack(pady=20)

label = tk.Label(root, text="Enter Message:", font=("Arial", 12))
label.pack()

entry = tk.Text(root, width=60, height=8)
entry.pack(pady=10)

button = tk.Button(
    root,
    text="Predict",
    font=("Arial", 12),
    command=predict
)
button.pack(pady=10)

output = tk.Label(root, text="", font=("Arial", 16, "bold"))
output.pack(pady=20)

root.mainloop()