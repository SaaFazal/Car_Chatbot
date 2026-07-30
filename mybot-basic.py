import aiml
import pandas as pd
import nltk
import string
import re
import csv
import os
import tensorflow as tf
import numpy as np
import tkinter as tk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sem import Expression
from nltk.inference import ResolutionProver
from colorama import init, Fore, Style
from tkinter import filedialog
from PIL import Image

# Initialize core settings
init(autoreset=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ensure required NLTK resources are available
for resource in ['wordnet', 'punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'corpora/{resource}.zip' if resource == 'wordnet' else f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

# Initialize AIML brain
kern = aiml.Kernel()
kern.verbose(False)
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

# --- Similarity Search Setup ---
qa_data = pd.read_csv("qa_pairs.csv")
questions = qa_data['question'].tolist()
answers = qa_data['answer'].tolist()
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Clean and lemmatize input text for matching."""
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = nltk.word_tokenize(text)
    return " ".join([lemmatizer.lemmatize(w) for w in words])

processed_questions = [preprocess_text(q) for q in questions]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_questions)

def get_similarity_response(user_input):
    """Find the best matching answer using TF-IDF and Cosine Similarity."""
    processed_input = preprocess_text(user_input)
    input_vector = vectorizer.transform([processed_input])
    similarities = cosine_similarity(input_vector, tfidf_matrix)[0]
    best_idx = similarities.argmax()
    if similarities[best_idx] > 0.7:
        return answers[best_idx]
    return "I'm sorry, I don't know the answer to that."

# --- Logical Reasoning System ---
read_expr = Expression.fromstring
kb_statements = []

def load_kb():
    """Load logical facts from CSV and verify integrity."""
    global kb_statements
    if not os.path.exists('logical-kb.csv'):
        return
    with open('logical-kb.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            expr = row['expression'].strip()
            if expr: kb_statements.append(read_expr(expr))
    
    # Verify integrity incrementally to avoid '-False' skolemization error
    temp_kb = []
    contradiction_found = False
    for stmt in kb_statements:
        negated = read_expr(f"-{stmt}")
        if ResolutionProver().prove(negated, temp_kb, verbose=False):
            contradiction_found = True
            break
        temp_kb.append(stmt)
    
    if contradiction_found:
        print(f"{Fore.RED}Warning: Knowledgebase contains contradictions!{Style.RESET_ALL}")
    else:
        print(f"{Fore.GREEN}Knowledgebase integrity verified.{Style.RESET_ALL}")

load_kb()

def check_fact(subject, relation):
    """Verify if a fact exists or can be inferred from the KB."""
    goal = read_expr(f"{relation}({subject})")
    if ResolutionProver().prove(goal, kb_statements, verbose=False):
        return "Correct"
    if ResolutionProver().prove(read_expr(f"-{relation}({subject})"), kb_statements, verbose=False):
        return "Incorrect"
    return "I don't know"

def add_fact(subject, relation):
    """Add a new fact to the KB after checking for contradictions."""
    expr_str = f"{relation}({subject})"
    new_expr = read_expr(expr_str)
    # Check if the negation of the new fact is already provable (contradiction)
    negated_expr = read_expr(f"-{expr_str}")
    if ResolutionProver().prove(negated_expr, kb_statements, verbose=False):
        return "Contradiction detected! Fact not added."
    
    kb_statements.append(new_expr)
    with open("logical-kb.csv", "a", newline='') as f:
        csv.writer(f).writerow([expr_str])
    return f"New fact added: {subject} is {relation}."

# --- Vision Module (VGG19) ---
cnn_model = None
if os.path.exists('models/car_logo_model.h5'):
    cnn_model = tf.keras.models.load_model('models/car_logo_model.h5')
    print("Vision system ready.")

def predict_image():
    """Handle image selection and classification."""
    if cnn_model is None:
        return "Vision system is offline."
        
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(initialdir="sample_images", title="Select Logo Image",
                                     filetypes=(("Images", "*.jpg *.jpeg *.png"), ("All", "*.*")))
    if not path: return "No image selected."
    
    print(f"{Fore.CYAN}Analyzing: {path}")
    try:
        img = Image.open(path).convert('RGB').resize((128, 128))
        arr = np.expand_dims(np.array(img).astype('float32') / 255.0, axis=0)
        preds = cnn_model.predict(arr, verbose=0)
        idx = np.argmax(preds[0])
        classes = ['a Hyundai', 'a Lexus', 'a Mazda', 'a Mercedes', 'an Opel', 'a Skoda', 'a Toyota', 'a Volkswagen']
        return f"Identification: {classes[idx]} (Confidence: {preds[0][idx]*100:.2f}%)"
    except Exception as e:
        return f"Classification error: {e}"

# --- Main Interaction Loop ---
print(f"{Fore.CYAN}\nCar Chatbot initialized. Type 'quit' to exit.\n")

while True:
    user_input = input(f"{Fore.YELLOW}You: {Style.RESET_ALL}").strip()
    if user_input.lower() == 'quit':
        print(f"{Fore.GREEN}Bot: Goodbye!")
        break
    
    # Priority 1: AIML rules
    response = kern.respond(user_input)
    if response == "CLASSIFY_IMAGE":
        print(f"{Fore.GREEN}Bot: Opening image selector...")
        print(f"{Fore.GREEN}Bot: {predict_image()}")
        continue
    elif response:
        print(f"{Fore.GREEN}Bot: {response}")
        continue
        
    # Priority 2: Logical Reasoning
    inp_lower = user_input.lower()
    add_match = re.match(r"i know that (\w+) is (\w+)", inp_lower)
    check_match = re.match(r"check that (\w+) is (\w+)", inp_lower)
    
    if add_match:
        s, r = add_match.groups()
        print(f"{Fore.GREEN}Bot: {add_fact(s.capitalize(), r.lower())}")
    elif check_match:
        s, r = check_match.groups()
        print(f"{Fore.GREEN}Bot: {check_fact(s.capitalize(), r.lower())}")
    else:
        # Priority 3: Similarity Fallback
        print(f"{Fore.GREEN}Bot: {get_similarity_response(user_input)}")
