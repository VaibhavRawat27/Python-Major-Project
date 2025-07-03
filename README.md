
# 🔐 Anonymous Feedback System

A secure, sentiment-aware feedback collection platform built using **Streamlit**, **NLTK**, and **Matplotlib**. Users can create feedback rooms, submit feedback anonymously, analyze sentiments, and download results – all without needing an account.

---

## 🚀 Features

- ✅ Create unique feedback rooms with a title and description
- ✍️ Submit anonymous feedback (with profanity and PII masking)
- 📊 Analyze feedback using:
  - Average sentiment (0–5 scale)
  - Sentiment pie chart (positive/neutral/negative)
  - Top keywords bar chart
  - AI-generated summaries (semantic and contextual)
- 🔼 Vote on feedback (upvote/downvote)
- 📥 Import & merge feedback from CSV
- 📤 Export feedback to CSV
- 🤖 Auto-summarization using keyword frequency and NLTK sentiment

---

## 📦 Dependencies

Install the required Python packages:

```bash
pip install streamlit pandas matplotlib nltk
```

NLTK will download resources at runtime:
```python
nltk.download("stopwords")
nltk.download("vader_lexicon")
```

---

## 🏁 How to Run

```bash
streamlit run main.py
```

> By default, Streamlit will launch on: `http://localhost:8501`

---

## 🗂️ Folder Structure

```
📁 rooms/                  # Stores all feedback CSVs and room metadata
├── [room_id].csv         # Feedback data
├── [room_id]_meta.csv    # Metadata (title + description)
📄 main.py                 # Streamlit app script
📄 README.md               # Project documentation
```

---

## 📌 Notes

- **Personal Info Masking**: Phone numbers and emails are masked.
- **Profanity Filter**: Basic offensive words are censored automatically.
- **Stateless Voting**: Votes are stored in CSV but not tied to IPs or users.
- **No Authentication**: Fully anonymous by design.

---

## 🙌 Built With

- [Streamlit](https://streamlit.io)
- [NLTK](https://www.nltk.org/)
- [Matplotlib](https://matplotlib.org/)
- 💡 Powered by your creativity!

---

## 📃 License

This project is open-source and free to use for educational and non-commercial purposes.

---

## 👨‍💻 Author

Made with ❤️ by Vaibhav Rawat
