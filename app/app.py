import streamlit as st
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, GRU, Dropout, Dense

# ==============================
# Constants
# ==============================
VOCAB_SIZE = 10000
MAX_LEN = 200

# ==============================
# Rebuild GRU Model Architecture
# ==============================
model = Sequential([
    Input(shape=(MAX_LEN,)),

    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=128
    ),

    GRU(64),

    Dropout(0.5),

    Dense(
        1,
        activation='sigmoid'
    )
])

# ==============================
# Load Trained Weights
# ==============================
model.load_weights("models/gru_weights.weights.h5")

# ==============================
# Load IMDb Word Dictionary
# ==============================
word_index = imdb.get_word_index()

word_index = {
    word: (index + 3)
    for word, index in word_index.items()
}

word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# ==============================
# Preprocessing Function
# ==============================
def preprocess_review(review):
    words = review.lower().split()

    encoded_review = []

    for word in words:
        encoded_review.append(
            word_index.get(word, 2)
        )

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    return padded_review


# ==============================
# Prediction Function
# ==============================
def predict_sentiment(review):
    processed_review = preprocess_review(review)

    prediction = model.predict(
        processed_review,
        verbose=0
    )[0][0]

    sentiment = (
        "Positive"
        if prediction >= 0.5
        else "Negative"
    )

    return sentiment, prediction


# ==============================
# Streamlit UI Config
# ==============================
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# ==============================
# UI
# ==============================
st.title("🎬 Movie Review Sentiment Analysis")

st.write(
    """
    This application predicts whether a movie review
    is Positive or Negative using a GRU Deep Learning model.
    """
)

review = st.text_area(
    "Enter your movie review:"
)

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning(
            "Please enter a review first."
        )

    else:
        sentiment, score = predict_sentiment(review)

        st.subheader("Prediction Result")

        st.write(
            f"Predicted Sentiment: **{sentiment}**"
        )

        st.write(
            f"Prediction Confidence Score: `{score:.4f}`"
        )

        if sentiment == "Positive":
            st.success(
                "This review is classified as Positive."
            )

        else:
            st.error(
                "This review is classified as Negative."
            )