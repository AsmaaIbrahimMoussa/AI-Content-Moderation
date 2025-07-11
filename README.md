# AI-Content-Moderation

This project provides an interactive **Streamlit-based application** that moderates both text and image inputs using a combination of a **CNN classifier**, **BLIP image captioning**, and **LLaMA Guard behavior** via the **Groq API**.


## Features

- **Text Moderation**
  - Detects toxic content using a trained **CNN model**.
  - Uses **LLaMA 3 (via Groq)** to classify input as "safe" or "unsafe".

- **Image Moderation**
  - Generates captions using **BLIP** from Hugging Face.
  - Sends the caption to LLaMA for moderation.
  - Uses CNN to classify image-related captions into categories.


## ModelS Overview

### 1. CNN Text Classifier
- Built with `TensorFlow` and `Keras`.
- Trained on a toxic content dataset using **class weighting** to address imbalance.
- Outputs probability scores for multiple classes like:
  - `safe`, `violent crime`, `self harm`, `child exploit sexual`, etc.

### 2. LLaMA Guard Behavior (Groq API)
- Simulates LLaMA Guard by prompting **LLaMA 3** with a strict moderation prompt (designed to enforce single-word safe/unsafe responses).

### 3. BLIP Integration
- Uses `Salesforce/blip-image-captioning-base` to convert images into descriptive captions.
- These captions are then moderated as text.


