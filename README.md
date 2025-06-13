
# 🎥 YouTube Video Q&A App

This is a Streamlit-based app that allows users to ask questions about the content of a YouTube video using its transcript. It uses LangChain, Gemini (Google Generative AI), OpenAI embeddings, and FAISS for semantic search.

---

## 🚀 Features

- 🔍 Enter a YouTube video ID
- 📄 Automatically fetch and process the video transcript
- ❓ Ask natural language questions about the video
- 🧠 Answers generated from transcript using Gemini LLM
- 🗂 Semantic chunking + vector search via FAISS
- 🔐 Secure API key management via `.env`

---

## 🧰 Tech Stack

- [Streamlit](https://streamlit.io/) — Web UI
- [LangChain](https://www.langchain.com/) — LLM pipelines
- [Gemini (Google Generative AI)](https://ai.google.dev) — LLM
- [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) — Text representation
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api) — Get video transcripts
- [python-dotenv](https://github.com/theskumar/python-dotenv) — Load environment variables

---

## 📦 Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/youtube-qa-app.git
   cd youtube-qa-app
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Add your API keys**

   Create a `.env` file in the root directory with the following:

   ```
   GOOGLE_API_KEY=your_google_api_key
   OPENAI_API_KEY=your_openai_api_key
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 🧪 Example Usage

1. Enter a YouTube video ID (e.g. `LPZh9BOjkQs`)
2. Click **"Proceed"** to load transcript
3. Ask a question like:
   - _"What was announced about Gemini in this video?"_
   - _"What is the main topic of the keynote?"_

---

## 📌 Notes

- The app only works with videos that have **available English transcripts**.
- No support for full URLs — only **video IDs** (e.g., `LPZh9BOjkQs`).
- Make sure your API keys have access to **Google Generative AI** and **OpenAI embeddings**.

---
