import streamlit as st
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load API keys from .env
load_dotenv()


# Page settings
st.set_page_config(page_title="YouTube Video Q&A")
st.title("🎥 YouTube Video Q&A App")

# Input Video ID
video_id = st.text_input("Enter YouTube Video ID (e.g., LPZh9BOjkQs):")

# Transcript placeholder
transcript = ""

# Step 1: Proceed button
if video_id and st.button("Proceed"):
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
        st.success("✅ Transcript loaded. You can now ask questions about the video.")
        st.session_state["transcript"] = transcript
    except TranscriptsDisabled:
        st.error("❌ No transcript available for this video.")
    except Exception as e:
        st.error(f"❌ Error fetching transcript: {e}")

# Step 2: Question input
if "transcript" in st.session_state:
    question = st.text_input("Ask a question about the video:")
    
    if question:
        with st.spinner("Thinking..."):

            # LLM & embeddings
            llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash')
            embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

            # Split transcript
            splitter = SemanticChunker(
                embeddings,
                breakpoint_threshold_type='standard_deviation',
                breakpoint_threshold_amount=0.5
            )
            chunks = splitter.create_documents([st.session_state["transcript"]])

            # Vector store & retriever
            vector_store = FAISS.from_documents(chunks, embeddings)
            retriever = vector_store.as_retriever(
                search_type='mmr',
                search_kwargs={'k': 4, 'lambda_mult': 0.5}
            )

            # Prompt setup
            prompt = PromptTemplate(
                template="""
                You are a helpful assistant.
                Answer ONLY from the provided transcript context.
                If the context is insufficient, just say you don't know.

                Context: {context}

                Question: {question}
                """,
                input_variables=['context', 'question']
            )

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)

            parser = StrOutputParser()
            parallel_chain = RunnableParallel({
                'context': retriever | RunnableLambda(format_docs),
                'question': RunnablePassthrough()
            })

            main_chain = parallel_chain | prompt | llm | parser
            answer = main_chain.invoke(question)

        st.markdown("### 💬 Answer")
        st.write(answer)

