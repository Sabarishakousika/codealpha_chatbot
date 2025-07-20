!pip install nltk scikit-learn

import nltk
nltk.download('punkt')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import ipywidgets as widgets
from IPython.display import display

# Expanded and cleaned training data
training_data = {
    "hello": "Hi there!",
    "hi": "Hello!",
    "hey": "Hey, how can I help you?",
    "how are you": "I'm doing well, thank you!",
    "bye": "Goodbye!",
    "goodbye": "See you soon!",
    "what is your name": "I'm CodeBot, your friendly assistant.",
    "what can you do": "I can chat, answer simple questions, and learn!",
    "thank you": "You're welcome!",
    "thanks": "No problem!",
    "good night": "Sweet dreams!",
    "help": "Sure, how can I assist you?"
}

questions = list(training_data.keys())
answers = list(training_data.values())

# Vectorize using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

# Response function using cosine similarity
def chatbot_response(user_input):
    user_input = user_input.lower()
    user_vec = vectorizer.transform([user_input])
    sim_scores = cosine_similarity(user_vec, X)
    max_score = np.max(sim_scores)

    if max_score < 0.3:
        return "🤔 Sorry, I didn’t understand that. Can you rephrase?"
    else:
        index = np.argmax(sim_scores)
        return answers[index]

# Widgets for input/output
input_box = widgets.Text(
    placeholder='Type your message here...',
    description='You:',
    layout=widgets.Layout(width='80%')
)
output_box = widgets.Output()

def on_enter_submit(text_widget):
    user_msg = text_widget.value
    if user_msg.strip().lower() == "exit":
        with output_box:
            print("Chat ended.")
        return
    response = chatbot_response(user_msg)
    with output_box:
        print(f"👤 You: {user_msg}")
        print(f"🤖 Bot: {response}\n")
    text_widget.value = ''

input_box.on_submit(on_enter_submit)

# Display chat UI
display(input_box, output_box)
