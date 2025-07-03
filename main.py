import streamlit as st
import pandas as pd
import os
import uuid
from datetime import datetime
import re
from collections import Counter
import nltk
from transformers import pipeline
import torch

# ─────────────────────────────────────────────────────────
# Lightweight NLTK setup (no punkt)
# ─────────────────────────────────────────────────────────
nltk.download("stopwords")
nltk.download("vader_lexicon")
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

# ─────────────────────────────────────────────────────────
# Constants & Config
# ─────────────────────────────────────────────────────────
DATA_DIR = "rooms"
os.makedirs(DATA_DIR, exist_ok=True)

st.set_page_config(page_title="🔐 Anonymous Feedback System", layout="wide")

# ─────────────────────────────────────────────────────────
# Summarization model (local)
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_summarizer():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="t5-small", tokenizer="t5-small", device=device)

summarizer = load_summarizer()

# ─────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────

def create_room(title: str, description: str = "") -> str:
    room_id = str(uuid.uuid4())[:8]
    pd.DataFrame(columns=["timestamp", "feedback", "upvotes", "downvotes"]).to_csv(
        os.path.join(DATA_DIR, f"{room_id}.csv"), index=False
    )
    pd.DataFrame([
        {"room_id": room_id, "title": title, "description": description}
    ]).to_csv(os.path.join(DATA_DIR, f"{room_id}_meta.csv"), index=False)
    return room_id

def _paths(room_id: str):
    return (
        os.path.join(DATA_DIR, f"{room_id}.csv"),
        os.path.join(DATA_DIR, f"{room_id}_meta.csv"),
    )

def get_room_data(room_id: str):
    fb_path, meta_path = _paths(room_id)
    if os.path.exists(fb_path) and os.path.exists(meta_path):
        return pd.read_csv(fb_path), pd.read_csv(meta_path).iloc[0], fb_path
    return None, None, None

def add_feedback(room_id: str, text: str):
    df, _, fb_path = get_room_data(room_id)
    if df is not None:
        new_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feedback": text,
            "upvotes": 0,
            "downvotes": 0,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(fb_path, index=False)

def vote_feedback(room_id: str, row_idx: int, direction: str):
    df, _, fb_path = get_room_data(room_id)
    if df is not None and row_idx in df.index:
        col = "upvotes" if direction == "up" else "downvotes"
        df.at[row_idx, col] += 1
        df.to_csv(fb_path, index=False)
        st.success(f"Feedback {row_idx + 1} {direction}voted!")  # Confirmation message

# ───────── NLP utilities ─────────

def rate_sentiment(texts):
    if not texts:
        return 0.0
    comp_scores = [sia.polarity_scores(t)["compound"] for t in texts]
    scaled = ((sum(comp_scores) / len(comp_scores)) + 1) / 2 * 5
    return round(scaled, 2)

def top_keywords(texts, n: int = 5):
    if not texts:
        return []
    words = re.findall(r"\b[a-zA-Z]{4,}\b", " ".join(texts).lower())
    freq = Counter(w for w in words if w not in STOPWORDS)
    return [kw for kw, _ in freq.most_common(n)]

def semantic_summary(texts):
    if not texts:
        return "No feedback yet."
    try:
        full_text = " ".join(texts)
        summary = summarizer(full_text[:1000], max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        return summary
    except Exception as e:
        return f"Unable to generate summary locally. Reason: {e}"

def stars_visual(rating: float) -> str:
    full = int(rating)
    half = 1 if rating - full >= 0.5 else 0
    empty = 5 - full - half
    return ("★" * full) + ("½" * half) + ("☆" * empty)

# ─────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────

st.title("🔐 Anonymous Feedback System")

create_tab, submit_tab, view_tab = st.tabs([
    "🆕 Create Room", "✍️ Submit Feedback", "📋 View Feedback"
])

with create_tab:
    st.subheader("Create a New Room")
    room_title = st.text_input("Room Title *")
    room_desc = st.text_area("Optional Description")

    if st.button("Create Room"):
        if room_title.strip():
            rid = create_room(room_title.strip(), room_desc.strip())
            st.success(f"Room created! ID: `{rid}`")
            st.code(f"http://localhost:8501/?room={rid}")
        else:
            st.warning("Room title required.")

with submit_tab:
    st.subheader("Submit Feedback")
    room_id_field = st.text_input("Room ID")
    feedback_field = st.text_area("Your Feedback")

    if st.button("Submit Feedback"):
        if room_id_field.strip() and feedback_field.strip():
            chk, _, _ = get_room_data(room_id_field.strip())
            if chk is not None:
                add_feedback(room_id_field.strip(), feedback_field.strip())
                st.success("Feedback submitted!")
            else:
                st.error("Room not found.")
        else:
            st.warning("Please provide both fields.")

with view_tab:
    st.subheader("View & Analyse Feedback")
    view_room_id = st.text_input("Room ID", key="vr")

    if view_room_id.strip():
        df, meta, _ = get_room_data(view_room_id.strip())

        if df is None:
            st.error("Invalid Room ID.")
        else:
            st.markdown(f"### {meta['title']}")
            if str(meta["description"]).strip():
                st.caption(meta["description"])

            feedback_texts = df["feedback"].tolist()

            if feedback_texts:
                with st.spinner("Calculating insights…"):
                    avg_rating = rate_sentiment(feedback_texts)
                    keywords = top_keywords(feedback_texts)
                    summary = semantic_summary(feedback_texts)

                st.metric("Average Sentiment", f"{avg_rating} / 5")
                st.markdown(stars_visual(avg_rating))
                st.markdown(f"**Top Keywords:** `{', '.join(keywords) if keywords else '—'}`")
                st.markdown("**Summary:**")
                st.info(summary)

            sort_cols = ["upvotes", "downvotes"]
            sorted_df = df.sort_values(sort_cols, ascending=[False, True])

            for order, (idx, row) in enumerate(sorted_df.iterrows(), start=1):
                col_text, col_up, col_down = st.columns([6, 1, 1])
                with col_text:
                    st.markdown(f"**Feedback {order}:** {row['feedback']}")
                    st.caption(f"🕒 {row['timestamp']}")
                with col_up:
                    if st.button(f"👍 {row['upvotes']}", key=f"up_{idx}"):
                        vote_feedback(view_room_id.strip(), idx, "up")
                        st.session_state.voted = True  # Set a flag to indicate voting occurred
                with col_down:
                    if st.button(f"👎 {row['downvotes']}", key=f"down_{idx}"):
                        vote_feedback(view_room_id.strip(), idx, "down")
                        st.session_state.voted = True  # Set a flag to indicate voting occurred

            # Check if voting occurred and refresh the data
            if 'voted' in st.session_state and st.session_state.voted:
                st.session_state.voted = False  # Reset the flag
                df, meta, _ = get_room_data(view_room_id.strip())  # Refresh data
                feedback_texts = df["feedback"].tolist()  # Update feedback texts

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Transformers (T5), and NLTK (VADER)")
