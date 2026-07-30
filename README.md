# Car Logo Classifier & Intelligent Assistant (uni-chatbot)

AI-powered web application integrating computer vision and natural language processing to assist users with automotive inquiries. 

## Features
- **Image Classification:** Fine-tuned ResNet-50 model deployed to classify car brand logos from user-uploaded images.
- **Conversational Agent:** Dual-agent NLP system utilizing NLTK/spaCy for deterministic intent routing, with a fallback to a generative LLM for open-ended queries.
- **Context-Aware Responses:** Maintains conversation context for follow-up questions.
- **Responsive UI:** Clean, chat-based web interface built for usability and speed.

## Tech Stack
- **Backend:** Python (Flask/FastAPI)
- **AI/ML:** PyTorch (ResNet-50), NLTK, spaCy, LLM Integration
- **Frontend:** HTML, CSS, JS
