# Car Logo Classifier & Intelligent Assistant (uni-chatbot)

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![NLP](https://img.shields.io/badge/NLP-NLTK%20%7C%20spaCy-green?style=for-the-badge)

An AI-powered web application that integrates computer vision and natural language processing to assist users with automotive inquiries. It serves as both an intelligent chatbot and a robust car brand identification tool.

## 🚀 Key Features

*   **Image Classification (Computer Vision):** Features a fine-tuned ResNet-50 Convolutional Neural Network (CNN) deployed to classify and identify car brand logos from user-uploaded images with high accuracy.
*   **Dual-Agent Conversational NLP:** Employs a hybrid NLP system utilizing NLTK and spaCy for fast, deterministic intent routing.
*   **Dynamic LLM Fallback:** For complex, open-ended, or out-of-domain queries, the system intelligently falls back to a generative Large Language Model (LLM) to construct dynamic responses.
*   **Context-Aware Dialog:** Maintains conversational context and memory across sessions to handle follow-up questions seamlessly.
*   **Responsive Web UI:** A clean, chat-based web interface built for high usability, drag-and-drop image uploads, and low-latency responses.

## 🛠️ Technology Stack

*   **Backend API:** Python (Flask/FastAPI)
*   **Computer Vision:** PyTorch, torchvision, ResNet-50
*   **Natural Language Processing:** NLTK, spaCy, HuggingFace Transformers
*   **Frontend UI:** HTML5, CSS3, Vanilla JavaScript

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SaaFazal/Car_Chatbot.git
   cd Car_Chatbot
   ```
2. **Environment Setup:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. **Download NLP Dependencies:**
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. **Run the Application:**
   ```bash
   python web_app.py
   ```
   Navigate to `http://localhost:5000` to interact with the assistant.

*(Note: The pre-trained ResNet-50 model `.h5` file and large datasets are excluded from this repository due to GitHub file size limits. You must run the training script or provide your own weights before starting the server).*
