# 🤖 NLP-Based Basic Chatbot | CodeAlpha Internship Task 4

This is a simple rule-based **Python chatbot** built using Natural Language Processing (NLP) techniques like **TF-IDF Vectorization** and **Cosine Similarity**. The chatbot can respond intelligently to basic user queries and improve response accuracy based on similarity scoring.

---

## 📁 Project Structure

- `chatbot.py` – Main Python script containing the chatbot logic
- `training_data.csv` – Data used to train and respond to user input
- `README.md` – Project overview and instructions

---

## 📌 Features

- Understands variations in greetings and questions
- Responds using closest matching known input
- Uses fallback response when confidence is low
- Interactive chat with a simple UI using `ipywidgets`

---

## 🔧 Technologies Used

- Python 3
- `nltk`
- `scikit-learn`
- `ipywidgets` (for interactive input in Jupyter or Colab)

---

## 🧠 How It Works

- User input is vectorized using **TF-IDF**
- Compared to pre-defined patterns using **cosine similarity**
- Best matching response is returned
- If similarity score is low, it returns a fallback message

---

## 🚀 Getting Started

1. Clone the repository or download the files.
2. Install dependencies:

```bash
pip install nltk scikit-learn ipywidgets
