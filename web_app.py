import streamlit as st
import aiml
import pandas as pd
import nltk
import string
import re
import csv
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.sem import Expression
from nltk.inference import ResolutionProver

# Set page configuration
st.set_page_config(
    page_title="Automotive Intelligent Assistant & Logo Classifier",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling for UI
st.markdown("""
<style>
    .reportview-container {
        background: #060f0e;
    }
    .stChatInput {
        border-radius: 20px;
    }
    .stProgress > div > div > div > div {
        background-color: #00f2fe;
    }
    .st-emotion-cache-1cv0e3m {
        border: 1px solid rgba(0, 242, 254, 0.2);
        background: rgba(17, 34, 32, 0.4);
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Ensure required NLTK resources are loaded
@st.cache_resource
def load_nltk_resources():
    for resource in ['wordnet', 'punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'corpora/{resource}.zip' if resource == 'wordnet' else f'tokenizers/{resource}')
        except LookupError:
            nltk.download(resource, quiet=True)

load_nltk_resources()

# Cache heavy ML and NLP resources for instant web interaction
@st.cache_resource
def load_aiml_kernel():
    kern = aiml.Kernel()
    kern.verbose(False)
    kern.setTextEncoding(None)
    kern.bootstrap(learnFiles="mybot-basic.xml")
    return kern

@st.cache_resource
def load_cnn_model():
    if os.path.exists('models/car_logo_model.h5'):
        return tf.keras.models.load_model('models/car_logo_model.h5')
    return None

@st.cache_resource
def load_similarity_engine():
    if os.path.exists("qa_pairs.csv"):
        qa_data = pd.read_csv("qa_pairs.csv")
        questions = qa_data['question'].tolist()
        answers = qa_data['answer'].tolist()
        
        lemmatizer = WordNetLemmatizer()
        
        def preprocess_text(text):
            text = text.translate(str.maketrans('', '', string.punctuation)).lower()
            words = nltk.word_tokenize(text)
            return " ".join([lemmatizer.lemmatize(w) for w in words])
            
        processed_questions = [preprocess_text(q) for q in questions]
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_questions)
        
        return vectorizer, tfidf_matrix, questions, answers, preprocess_text
    return None, None, [], [], None

# Initialize resources
kern = load_aiml_kernel()
cnn_model = load_cnn_model()
vectorizer, tfidf_matrix, questions, answers, preprocess_fn = load_similarity_engine()

# --- Symbolic Logic Core ---
read_expr = Expression.fromstring

def get_kb_statements():
    """Load active logical facts from CSV."""
    statements = []
    if not os.path.exists('logical-kb.csv'):
        return statements
    with open('logical-kb.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            expr = row['expression'].strip()
            if expr: 
                try:
                    statements.append(read_expr(expr))
                except Exception:
                    pass
    return statements

def verify_kb_integrity(statements):
    """Check if the current logical KB is consistent."""
    temp_kb = []
    for stmt in statements:
        negated = read_expr(f"-{stmt}")
        if ResolutionProver().prove(negated, temp_kb, verbose=False):
            return False
        temp_kb.append(stmt)
    return True

def add_logical_fact(subject, relation):
    """Attempt to add a new logical fact, validating against contradictions."""
    expr_str = f"{relation}({subject})"
    try:
        new_expr = read_expr(expr_str)
        negated_expr = read_expr(f"-{expr_str}")
        
        current_statements = get_kb_statements()
        
        # Check if the opposite is already logically provable (contradiction)
        if ResolutionProver().prove(negated_expr, current_statements, verbose=False):
            return False, "Contradiction detected! This statement conflicts with stored knowledge."
            
        # Append fact to local file
        with open("logical-kb.csv", "a", newline='') as f:
            csv.writer(f).writerow([expr_str])
            
        return True, f"Knowledge extended successfully: {subject} is verified as {relation}."
    except Exception as e:
        return False, f"Logic parse error: {e}"

def check_logical_fact(subject, relation):
    """Verify if a fact can be logically proven or disproven."""
    try:
        current_statements = get_kb_statements()
        goal = read_expr(f"{relation}({subject})")
        negated_goal = read_expr(f"-{relation}({subject})")
        
        if ResolutionProver().prove(goal, current_statements, verbose=False):
            return "Correct"
        if ResolutionProver().prove(negated_goal, current_statements, verbose=False):
            return "Incorrect"
        return "Insufficient Information"
    except Exception as e:
        return f"Prover error: {e}"

# --- Title Header ---
st.title("🚗 Car Logo Classifier & Intelligent Assistant")
st.markdown("An advanced AI showcase unifying **Deep Learning Vision (VGG19)**, **First-Order Symbolic Logic Proving**, and **Statistical NLP Search**.")
st.write("---")

# Setup layout tabs
tab1, tab2 = st.tabs(["💬 Assistant Conversation", "📷 Logo Computer Vision"])

# --- Sidebar Logic KB Manager ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/car.png", width=70)
    st.header("🧠 Symbolic Knowledge Base")
    st.caption("Active facts grounded in First-Order Logic (FOL)")
    
    # FOL Facts List
    kb_facts = get_kb_statements()
    if kb_facts:
        facts_df = pd.DataFrame({"FOL Expression": [str(f) for f in kb_facts]})
        st.dataframe(facts_df, use_container_width=True, height=200)
    else:
        st.info("No logical facts stored in KB.")
        
    st.write("---")
    
    # Learning Form
    st.subheader("💡 Teach the Assistant")
    with st.form("learning_form", clear_on_submit=True):
        subject = st.text_input("Subject (e.g. Tesla, Leaf)", placeholder="Car model/brand").strip()
        relation = st.text_input("Relation (e.g. Electric, Petrol)", placeholder="Car property").strip()
        submit_btn = st.form_submit_button("Assert Fact to KB")
        
        if submit_btn:
            if subject and relation:
                success, msg = add_logical_fact(subject.capitalize(), relation.lower())
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.warning("Please fill out both inputs.")
                
    st.write("---")
    
    # Logic Checker Form
    st.subheader("🔍 Query Logical Truth")
    with st.form("query_form"):
        q_subject = st.text_input("Subject", placeholder="e.g. Tesla").strip()
        q_relation = st.text_input("Relation", placeholder="e.g. electric").strip()
        check_btn = st.form_submit_button("Run Resolution Prover")
        
        if check_btn:
            if q_subject and q_relation:
                result = check_logical_fact(q_subject.capitalize(), q_relation.lower())
                if result == "Correct":
                    st.success("✅ **Correct:** This fact is logically TRUE.")
                elif result == "Incorrect":
                    st.error("❌ **Incorrect:** This fact is logically FALSE (contradicts knowledge).")
                else:
                    st.info("❓ **Insufficient Info:** The prover cannot prove or disprove this fact.")
            else:
                st.warning("Please fill out both inputs.")

# --- Tab 1: Stateful Assistant Chat ---
with tab1:
    st.subheader("Dialogue Center")
    
    # Initialize message session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Welcome! I am your Car Assistant. I can chat, prove logical queries, or classify uploaded car brand logos. How can I help you today?"}
        ]
        
    # Render messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Process incoming inputs
    if user_input := st.chat_input("Ask a car question (or assert logic with 'i know that [car] is [property]')"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Determine Response
        bot_response = ""
        
        # Priority 1: AIML pattern matching
        aiml_res = kern.respond(user_input)
        if aiml_res == "CLASSIFY_IMAGE":
            bot_response = "To identify a car logo image, please navigate to the **'Logo Computer Vision'** tab above, upload your image file, and let the VGG19 CNN perform real-time emblem classification."
        elif aiml_res:
            bot_response = aiml_res
            
        # Priority 2: Logical Reasoning Commands
        if not bot_response:
            inp_lower = user_input.lower().strip()
            add_match = re.match(r"i know that (\w+) is (\w+)", inp_lower)
            check_match = re.match(r"check that (\w+) is (\w+)", inp_lower)
            
            if add_match:
                s, r = add_match.groups()
                success, msg = add_logical_fact(s.capitalize(), r.lower())
                bot_response = msg
                if success:
                    st.rerun() # Refresh sidebar facts
            elif check_match:
                s, r = check_match.groups()
                result = check_logical_fact(s.capitalize(), r.lower())
                if result == "Correct":
                    bot_response = f"Correct. The Resolution Prover confirms {s.capitalize()} is indeed {r.lower()}."
                elif result == "Incorrect":
                    bot_response = f"Incorrect. Stored rules contradict the statement that {s.capitalize()} is {r.lower()}."
                else:
                    bot_response = f"I don't have enough logical information in my database to verify if {s.capitalize()} is {r.lower()}."
                    
        # Priority 3: TF-IDF Cosine Similarity Fallback
        if not bot_response and vectorizer is not None:
            processed_input = preprocess_fn(user_input)
            input_vector = vectorizer.transform([processed_input])
            similarities = cosine_similarity(input_vector, tfidf_matrix)[0]
            best_idx = similarities.argmax()
            if similarities[best_idx] > 0.7:
                bot_response = answers[best_idx]
                
        # Priority 4: Default Fallback
        if not bot_response:
            bot_response = "I'm sorry, I don't know the answer to that. You can try teaching me about this car, or rephrase your question."
            
        # Display bot response
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

# --- Tab 2: VGG19 CNN Classifier UI ---
with tab2:
    st.subheader("Keras VGG19 Convolutional Neural Network Classifier")
    st.markdown("Upload a brand logo image to run a forward propagation pass through the fine-tuned VGG19 model.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Upload Logo Image (.jpg, .jpeg, .png)", type=["jpg", "jpeg", "png"])
        
        # Sample selection helper
        st.markdown("💡 **No images on hand?** Select a test emblem from the preloaded dataset:")
        sample_emblem = st.selectbox("Pick a Sample Image", [
            "None",
            "sample_images/skoda_logo.jpg",
            "sample_images/mercedes_logo.jpg",
            "sample_images/toyota_logo.png",
            "sample_images/volkswagen_logo.png",
            "sample_images/opel_logo.jpg"
        ])
        
        # Determine image source
        img_to_classify = None
        if uploaded_file is not None:
            img_to_classify = Image.open(uploaded_file)
            st.success("File uploaded successfully.")
        elif sample_emblem != "None" and os.path.exists(sample_emblem):
            img_to_classify = Image.open(sample_emblem)
            st.info(f"Loaded sample image: {sample_emblem}")
            
    with col2:
        if img_to_classify is not None:
            st.image(img_to_classify, caption="Target Emblem for Classification", width=220)
            
            classify_btn = st.button("Run Neural Network Inference", type="primary")
            if classify_btn:
                if cnn_model is not None:
                    with st.spinner("Executing forward pass through VGG19..."):
                        # Preprocess image precisely to match model bounds (128x128 normalized RGB)
                        img_resized = img_to_classify.convert('RGB').resize((128, 128))
                        arr = np.expand_dims(np.array(img_resized).astype('float32') / 255.0, axis=0)
                        
                        # Inference
                        preds = cnn_model.predict(arr, verbose=0)
                        idx = np.argmax(preds[0])
                        confidence = preds[0][idx] * 100
                        
                        classes = ['Hyundai', 'Lexus', 'Mazda', 'Mercedes', 'Opel', 'Skoda', 'Toyota', 'Volkswagen']
                        predicted_brand = classes[idx]
                        
                        # Display Results beautifully
                        st.write("---")
                        st.subheader("📊 Network Prediction Results")
                        st.metric(label="Identified Brand Logo", value=predicted_brand)
                        st.write(f"**Classification Confidence:** {confidence:.2f}%")
                        st.progress(int(confidence))
                else:
                    st.error("Deep learning model file 'models/car_logo_model.h5' was not detected in workspace. Classification is offline.")
        else:
            st.info("Please upload an image or select a sample brand logo to activate deep learning classification.")
