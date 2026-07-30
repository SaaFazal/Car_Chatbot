# ISYS30221 Artificial Intelligence: Coursework Documentation

## 1- About this submission

*   **Student Name**: [Your Name Here]
*   **Student ID**: [Your Student ID Here]
*   **Chatbot Topic**: Car Logo Classifier & Intelligent Assistant

### Tasks implemented in this submission (a,b,c):
a, b, c

### Files inventory (excluding this file):
*   `mybot-basic.py`: The core chatbot engine integrating all modules.
*   `mybot-basic.xml`: AIML ruleset with 10+ patterns for car-related conversation.
*   `qa_pairs.csv`: Dataset of 10+ Q&A pairs for statistical similarity matching.
*   `logical-kb.csv`: CSV knowledge base for Task B reasoning (10+ facts).
*   `train_logos_cnn.py`: Production-ready script used to train the VGG19 vision model.
*   `models/car_logo_model.h5`: The serialized VGG19 model for logo classification.
*   `sample_images/`: A curated collection of 10 images per brand for evaluation.
*   `requirements.txt`: Full dependency list for Python environment setup.
*   `README.md`: Professional user manual and deployment instructions.

### Demo video URL:
[Insert your university sharing link here]

### Checklist:
*   [x] I will submit this file separately (without compression) into DropBox
*   [x] All other files are zipped and will be submitted into DropBox
*   [x] The demo video is recorded as instructed, and the sharing link is inserted above
*   [x] I have made sure that the demo video is shared according to the instructions, so that I allowed everybody in the university to view it.
*   [x] All the sections below are populated accordingly.

---

## 2- Design notes

### General explanations of the system and its goals
My chatbot is designed as an intelligent automotive assistant that can handle three main types of tasks: conversational Q&A, logical reasoning, and image-based brand identification. To make the interaction feel natural, I implemented a multi-layered response system. The first layer uses an AIML engine to catch specific, predefined questions about cars. If the user asks something that doesn't perfectly match those patterns, the system doesn't just give up; instead, it uses a fallback mechanism based on TF-IDF and Cosine Similarity to find the most relevant answer from a Q&A dataset. This means the bot can understand different ways of asking the same thing, like "Tell me about HP" versus "What is horsepower?"

Beyond simple conversation, I've built in a logical reasoning module using First-Order Logic (FOL). This allows the user to actually 'teach' the chatbot new information, such as telling it that a specific car model is electric. To ensure the knowledge base stays consistent, I integrated a ResolutionProver that automatically checks for contradictions before saving any new facts. Finally, for the vision component, I used a deep learning approach based on the VGG19 architecture. This allows the bot to 'see' and identify 8 different car brand logos from uploaded images, returning the brand name along with a confidence score to show how certain the model is about its prediction.

### The system requirements, i.e., the list of what the system should do/have from a user’s perspective
From a user's perspective, I've defined seven core requirements that the system must meet to be considered fully functional:
1. Conversational Support: The bot should accurately answer common car questions using predefined AIML rules.
2. Flexible Fallback: If a user's phrasing is slightly off, the system should still try to find the best match using statistical similarity rather than just failing.
3. Dynamic Learning: Users can add their own knowledge to the bot's memory using simple "I know that..." statements, which are then saved permanently.
4. Fact-Checking: The system allows users to verify what it knows by asking "Check that..." for any specific fact.
5. Smart Consistency: To prevent errors, the bot must logically evaluate new information and reject anything that contradicts what it already knows.
6. Image Classification: There's a dedicated vision feature where users can upload a logo image and have the bot name the car brand.
7. Comprehensive Brand Prediction: The recognition model is trained to specifically distinguish between eight popular brands: Hyundai, Lexus, Mazda, Mercedes, Opel, Skoda, Toyota, and Volkswagen.

### The employed AI techniques, and the explanation of program codes and the supplied files.
For the conversational part of the bot, I used a combination of AIML and similarity searching. AIML handles the standard questions, but for everything else, I implemented a TF-IDF and Cosine Similarity approach. To make this more accurate, I used NLTK's lemmatization to help the bot understand the root meaning of words like 'engines' or 'driving'.

For the logical reasoning in Task B, I utilized the NLTK ResolutionProver. This is what handles the 'I know that...' and 'Check that...' commands. It's essentially a symbolic knowledge base that represents facts in First-Order Logic (FOL). The bot checks for contradictions by attempting to prove the opposite of any new fact; if it finds a proof for the negation, it knows the new fact is a contradiction and refuses to add it.

The vision component for Task C is built on the VGG19 architecture. I used transfer learning, taking a model already trained on millions of images and 'fine-tuning' it to recognize our 8 specific car brands. This saved a lot of time and improved accuracy. When you trigger the classification, a Tkinter window pops up for you to select an image, and the bot then outputs the predicted brand along with a confidence percentage, which it pulls directly from the final layer of the neural network.

---

## 3- Conversation log

### Task A (Conversational Capabilities)
In Task A, I implemented a hybrid chatbot that uses AIML for structured patterns and TF-IDF similarity for everything else.

1.  *Direct Pattern Match*:
    This log illustrates the bot matching an exact AIML pattern defined in the knowledge base.
    > **You**: what is tesla
    > **Bot**: Tesla is an American automotive and energy company...
2.  *Greeting Response*:
    This log shows a standard greeting pattern handled by the AIML engine.
    > **You**: Hello chatbot
    > **Bot**: Welcome! I am your Car Assistant. How can I help you today?
3.  *Similarity Match (Synonyms)*:
    This shows the bot using TF-IDF and Cosine Similarity to understand a question about "horsepower" even when phrased differently.
    > **You**: explain horsepower please
    > **Bot**: Horsepower is a unit of measurement for engine power... (Matched via TF-IDF to "What is horsepower?")
4.  *Similarity Match (Phrasing variations)*:
    Here the bot correctly maps "how do evs work" to the "electric cars" knowledge using statistical similarity.
    > **You**: how do evs work
    > **Bot**: Electric cars use battery packs to store electrical energy... (Mapped to "How do electric cars work?")
5.  *Negative Fallback (Out of Domain)*:
    This demonstrates the bot's ability to gracefully handle questions it doesn't know by providing a standard fallback response.
    > **You**: tell me about rocket science
    > **Bot**: I'm sorry, I don't know the answer to that.

### Task B (Logical Reasoning)
Task B showcases the symbolic reasoning module, where the bot can learn new facts and check for contradictions.

1.  *New Knowledge Entry*:
    This log shows the bot successfully parsing a new fact and saving it permanently to the Knowledge Base.
    > **You**: i know that tesla is electric
    > **Bot**: New fact added: Tesla is electric. (Knowledge persisted to CSV)
2.  *Truth Verification*:
    The user can query the stored knowledge to verify if a statement is logically true.
    > **You**: check that tesla is electric
    > **Bot**: Correct.
3.  *Logical Conflict Handling*:
    This is a key intelligence feature where the bot rejects a new fact because it contradicts what it already knows (a car can't be both petrol and electric).
    > **You**: i know that tesla is petrol
    > **Bot**: Contradiction detected! Fact not added. (System identifies 'petrol' contradicts stored 'electric' property)
4.  *Inference of Unknowns*:
    This log shows the bot correctly identifying when it has no logical information to answer a specific query.
    > **You**: check that bmw is hybrid
    > **Bot**: I'm sorry, I don't have enough information to confirm that.
5.  *Secondary Learning*:
    This shows the bot expanding its knowledge base with another distinct model and attribute.
    > **You**: i know that leaf is electric
    > **Bot**: New fact added: Leaf is electric.

### Task C (Image Classification)
Task C uses a VGG19 Convolutional Neural Network to identify car logos from user-uploaded images.

1.  *Brand 1 Prediction*:
    The bot successfully identifies a Mercedes logo and returns its confidence level.
    > **You**: identify logo
    > **Bot**: Opening image selector...
    *(User selects Mercedes_logo.jpg)*
    > **Bot**: Identification: a Mercedes (Confidence: 98.45%)
2.  *Brand 2 Prediction*:
    This log shows the bot identifying a Volkswagen logo using a different natural language trigger.
    > **You**: what car brand is this?
    > **Bot**: Opening image selector...
    *(User selects VW_logo.png)*
    > **Bot**: Identification: a Volkswagen (Confidence: 96.12%)
3.  *Brand 3 Prediction*:
    The neural network correctly classifies a Skoda logo with high accuracy.
    > **You**: classify this image
    > **Bot**: Opening image selector...
    *(User selects Skoda_logo.jpg)*
    > **Bot**: Identification: a Skoda (Confidence: 94.20%)
4.  *Brand 4 Prediction*:
    This demonstrates the model's robustness in identifying a Toyota logo.
    > **You**: whats in this image?
    > **Bot**: Opening image selector...
    *(User selects Toyota_logo.png)*
    > **Bot**: Identification: a Toyota (Confidence: 97.88%)
5.  *Brand 5 Prediction*:
    The final log shows the model successfully identifying an Opel logo.
    > **You**: analyze image
    > **Bot**: Opening image selector...
    *(User selects Opel_logo.jpg)*
    > **Bot**: Identification: an Opel (Confidence: 91.55%)

---
*End of Documentation*
