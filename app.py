import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import os 
from dotenv import load_dotenv

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Load CNN model and tokenizer
@st.cache_resource
def load_all_models():
    model = load_model("cnn_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

cnn_model, tokenizer, label_encoder = load_all_models()

# Load BLIP
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("blip_model")
    blip_model = BlipForConditionalGeneration.from_pretrained("blip_model")
    return processor, blip_model

blip_processor, blip_model = load_blip_model()


load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def llama_guard_check(text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": "llama3-8b-8192", 
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are LLaMA Guard, a content moderation model. "
                    "Classify the following text as either 'safe' or 'unsafe'. "
                    "Only respond with one word: 'safe' or 'unsafe'."
                )
            },
            {"role": "user", "content": text}
        ],
        "temperature": 0.0
    }
    response = requests.post(url, headers=headers, json=body)
    return response.json()["choices"][0]["message"]["content"].strip().lower()

def generate_caption(image):
    inputs = blip_processor(images=image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

def is_safe_by_llama(prompt):
    try:
        response = llama_guard_check(prompt)
        if response == "safe":
            return True, "✅ The content is SAFE."
        else:
            return False, "🚫 The content is UNSAFE."
    except Exception as e:
        return False, f"⚠️ Error during safety check: {str(e)}"

def clean_and_preprocess(text):
    text = re.sub("[^a-zA-Z]", " ", str(text)).lower().strip()
    tokens = nltk.word_tokenize(text)
    return " ".join([stemmer.stem(w) for w in tokens if w not in stop_words])

def predict_toxic_category(text):
    text = clean_and_preprocess(text)
    max_len = 100
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    probs = cnn_model.predict(padded)[0]
    classes = label_encoder.classes_
    top_idx = np.argmax(probs)
    results = {classes[i]: float(probs[i]) for i in range(len(classes))}
    return classes[top_idx], results

# Streamlit UI
st.title("AI Content Moderation")
st.set_page_config(
    page_title="AI Content Moderation",  
    layout="centered",                  
    initial_sidebar_state="auto"
)


input_type = st.radio("Choose input type:", ["Text", "Image"])

if input_type == "Text":
    user_text = st.text_area("Enter your text:")
    if st.button("Moderate Text"):
        if user_text.strip():
            is_safe, response = is_safe_by_llama(user_text)
            if not is_safe:
                st.error(response)
            else:
                st.success(response)
                category, probs = predict_toxic_category(user_text)
                st.write(f"### Predicted Category: `{category}`")
                st.bar_chart(probs)
        else:
            st.warning("Please enter some text.")

else:
    uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        caption = generate_caption(image)
        st.write(f"📷 **Image Caption (BLIP):** `{caption}`")

        is_safe, response = is_safe_by_llama(caption)
        if not is_safe:
            st.error(response)
        else:
            st.success(response)
            category, probs = predict_toxic_category(caption)
            st.write(f"### Predicted Category: `{category}`")
            st.bar_chart(probs)
