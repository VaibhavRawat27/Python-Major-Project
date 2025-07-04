import streamlit as st
import pandas as pd
import os
import uuid
import re
from datetime import datetime
from collections import Counter
import nltk
import csv
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# ───────── NLTK Setup ─────────
nltk.download("stopwords")
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()
STOPWORDS = set(nltk.corpus.stopwords.words("english"))

# ───────── App Config ─────────
DATA_DIR = "rooms"
os.makedirs(DATA_DIR, exist_ok=True)
st.set_page_config(page_title="🔐 Smart Feedback Analyzer", layout="wide")

# ───────── Helper Functions ─────────
def create_room(title, description=""):
    room_id = str(uuid.uuid4())[:8]
    pd.DataFrame(columns=["timestamp", "feedback", "upvotes", "downvotes"]).to_csv(f"{DATA_DIR}/{room_id}.csv", index=False, quoting=csv.QUOTE_ALL)
    pd.DataFrame([{"room_id": room_id, "title": title, "description": description}]).to_csv(f"{DATA_DIR}/{room_id}_meta.csv", index=False)
    return room_id

def _paths(room_id):
    return f"{DATA_DIR}/{room_id}.csv", f"{DATA_DIR}/{room_id}_meta.csv"

def get_room_data(room_id):
    fb_path, meta_path = _paths(room_id)
    try:
        df = pd.read_csv(fb_path, quoting=csv.QUOTE_MINIMAL)
        meta = pd.read_csv(meta_path).iloc[0]
        return df, meta, fb_path
    except Exception as e:
        st.warning(f"Error loading room data: {e}")
        return None, None, None

def mask_personal_info(text):
    text = re.sub(r'\b\d{10}\b', '***', text)
    text = re.sub(r'\S+@\S+', '***', text)
    bad_words = ['fuck', 'bastard', 'asshole', 'idiot']
    for word in bad_words:
        text = re.sub(rf'\b{word}\b', '***', text, flags=re.IGNORECASE)
    return text

def add_feedback(room_id, text):
    df, _, fb_path = get_room_data(room_id)
    if df is not None:
        new_row = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "feedback": mask_personal_info(text),
            "upvotes": 0,
            "downvotes": 0,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(fb_path, index=False, quoting=csv.QUOTE_ALL)

def vote_feedback(room_id, row_idx, direction):
    df, _, fb_path = get_room_data(room_id)
    if df is not None and 0 <= row_idx < len(df):
        col = "upvotes" if direction == "up" else "downvotes"
        try:
            df.at[row_idx, col] = int(df.at[row_idx, col]) + 1
            df.to_csv(fb_path, index=False, quoting=csv.QUOTE_ALL)
        except Exception as e:
            st.error(f"Vote error: {e}")

def rate_sentiment(texts):
    if not texts:
        return 0.0
    scores = [sia.polarity_scores(t)["compound"] for t in texts]
    scaled = ((sum(scores) / len(scores)) + 1) / 2 * 5
    return round(scaled, 2)

def display_star_rating(score):
    full_stars = int(score)
    half_star = score - full_stars >= 0.5
    stars = "⭐" * full_stars
    if half_star:
        stars += "✨"
    return stars.ljust(5, "☆")

def top_keywords(texts, n=5):
    words = re.findall(r"\b[a-zA-Z]{4,}\b", " ".join(texts).lower())
    freq = Counter(w for w in words if w not in STOPWORDS)
    return freq.most_common(n)

def semantic_summary(texts):
    full_text = " ".join(texts)
    positive_words = [w for w in full_text.split() if sia.polarity_scores(w)["compound"] > 0.05]
    negative_words = [w for w in full_text.split() if sia.polarity_scores(w)["compound"] < -0.05]
    rating = rate_sentiment(texts)

    if rating >= 4: summary = "😊 People are very happy."
    elif rating >= 3: summary = "🙂 People are mostly satisfied."
    elif rating >= 2: summary = "😐 Mixed feelings."
    elif rating >= 1: summary = "😞 Mostly unhappy."
    else: summary = "😡 Very upset."

    return f"**Positive words:** {len(positive_words)}\n**Negative words:** {len(negative_words)}\n\n{summary}"

def plot_sentiment_pie(texts):
    counts = [0, 0, 0]
    for t in texts:
        score = sia.polarity_scores(t)["compound"]
        if score > 0.05: counts[0] += 1
        elif score < -0.05: counts[2] += 1
        else: counts[1] += 1
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.pie(counts, labels=["Positive", "Neutral", "Negative"], autopct='%1.1f%%', colors=["green", "gray", "red"])
    return fig

def plot_top_keywords_bar(keywords):
    if not keywords: return None
    words, counts = zip(*keywords)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.barh(words, counts, color='skyblue')
    ax.invert_yaxis()
    return fig

def summarize_feedback_ai(texts):
    if not texts:
        return "No feedback to summarize."

    from heapq import nlargest
    positive_feedback = []
    negative_feedback = []
    neutral_feedback = []

    for text in texts:
        score = sia.polarity_scores(text)["compound"]
        if score > 0.05:
            positive_feedback.append(text)
        elif score < -0.05:
            negative_feedback.append(text)
        else:
            neutral_feedback.append(text)

    def extract_key_sentences(feedback_list, num_sentences=2):
        if not feedback_list:
            return ""
        combined_text = " ".join(feedback_list)
        sentences = re.split(r'(?<=[.!?])\s+', combined_text)
        words = re.findall(r'\b\w{4,}\b', combined_text.lower())
        freq = Counter(w for w in words if w not in STOPWORDS)
        sentence_scores = {
            sentence: sum(freq[word] for word in freq if word in sentence.lower())
            for sentence in sentences if len(sentence.split()) > 3
        }
        summary_sentences = nlargest(num_sentences, sentence_scores, key=sentence_scores.get)
        return " ".join(summary_sentences)

    parts = []
    if (ps := extract_key_sentences(positive_feedback)):
        parts.append(f"**Some users expressed positive sentiment, highlighting:** {ps}")
    if (ns := extract_key_sentences(negative_feedback)):
        parts.append(f"**On the other hand, some users reported concerns such as:** {ns}")
    if (neutral_summary := extract_key_sentences(neutral_feedback, 1)) and not parts:
        parts.append(f"**Some neutral feedback observed:** {neutral_summary}")
    return "\n\n".join(parts) if parts else "Summary could not be generated."

# ───────── Streamlit UI ─────────
st.title("🔐 Smart Feedback Analyzer")
tab1, tab2, tab3, tab4 = st.tabs(["🆕 Create Room", "✍️ Submit Feedback", "📋 View Feedback", "📁 Import File"])

with tab1:
    st.subheader("Create New Room")
    title = st.text_input("Room Title *", key="create_title")
    desc = st.text_area("Description", key="create_desc")
    if st.button("Create Room", key="create_btn") and title.strip():
        rid = create_room(title, desc)
        st.success(f"Room ID: `{rid}`")
        st.code(f"http://localhost:8501/?room={rid}")

with tab2:
    st.subheader("Submit Feedback")
    rid = st.text_input("Room ID", key="submit_room_id")
    text = st.text_area("Your Feedback", key="submit_feedback")
    if st.button("Submit Feedback", key="submit_btn") and rid.strip() and text.strip():
        chk, _, _ = get_room_data(rid)
        if chk is not None:
            add_feedback(rid, text)
            st.success("Feedback submitted!")
        else:
            st.error("Room not found.")

with tab3:
    st.subheader("View & Analyze")
    view_id = st.text_input("Room ID", key="view_room_id")
    if view_id.strip():
        df, meta, path = get_room_data(view_id)
        if df is not None:
            st.markdown(f"### {meta['title']}")
            st.caption(meta['description'])
            feedbacks = df["feedback"].astype(str).tolist()

            if feedbacks:
                rating = rate_sentiment(feedbacks)
                stars = display_star_rating(rating)
                keywords = top_keywords(feedbacks, 10)

                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric("Avg Sentiment", f"{rating}/5")
                with col2:
                    st.markdown(f"**Rating:** {stars}")

                st.markdown(semantic_summary(feedbacks))

                col1, col2 = st.columns(2)
                with col1:
                    st.pyplot(plot_sentiment_pie(feedbacks))
                with col2:
                    chart = plot_top_keywords_bar(keywords)
                    if chart:
                        st.pyplot(chart)
                    else:
                        st.info("No keyword data.")

                st.subheader("🧠 AI Summary")
                st.info(summarize_feedback_ai(feedbacks))

            st.subheader("📤 Export")
            st.download_button("Download CSV", df.to_csv(index=False, quoting=csv.QUOTE_ALL).encode(), f"{view_id}_feedback.csv")

            st.subheader("📌 Feedbacks")
            sorted_df = df.sort_values(["upvotes", "downvotes"], ascending=[False, True]).reset_index()

            for i, row in sorted_df.iterrows():
                col1, col2, col3 = st.columns([6, 1, 1])
                with col1:
                    st.markdown(f"**{row['feedback']}**")
                    st.caption(row['timestamp'])
                with col2:
                    if st.button(f"👍 {row['upvotes']}", key=f"up_{i}"):
                        vote_feedback(view_id, row['index'], "up")
                        st.success("Upvoted!")
                with col3:
                    if st.button(f"👎 {row['downvotes']}", key=f"down_{i}"):
                        vote_feedback(view_id, row['index'], "down")
                        st.success("Downvoted!")
        else:
            st.error("Invalid Room ID.")

with tab4:
    st.subheader("Upload Feedback CSV")
    import_id = st.text_input("Room ID", key="import_room_id")
    file = st.file_uploader("Upload CSV (timestamp, feedback, upvotes, downvotes)", type=["csv"], key="import_file")
    if file and import_id:
        try:
            new_data = pd.read_csv(file)
            if all(col in new_data.columns for col in ["timestamp", "feedback", "upvotes", "downvotes"]):
                existing, _, path = get_room_data(import_id)
                if existing is not None:
                    combined = pd.concat([existing, new_data], ignore_index=True)
                    combined.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
                    st.success("Uploaded & merged successfully.")
                else:
                    st.error("Room not found.")
            else:
                st.error("Invalid CSV format.")
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, NLTK, and Matplotlib")
