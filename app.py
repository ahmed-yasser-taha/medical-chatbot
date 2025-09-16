import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def main():
    # ====== واجهة العنوان + ترحيب ======
    st.markdown(
        """
        <style>
        .title { 
            font-size: 2.2em; 
            font-weight: bold; 
            color: #4CAF50; 
            text-align: center; 
        }
        .subtitle { 
            font-size: 1.1em; 
            color: #555; 
            text-align: center; 
            margin-bottom: 20px; 
        }
        </style>
        """, unsafe_allow_html=True
    )

    st.markdown('<p class="title">🤖 Ask Chatbot!</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Welcome! This chatbot answers your questions using trusted reference documents. Just type below ⬇️</p>', unsafe_allow_html=True)

    # ====== الرسائل ======
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    # ====== إدخال المستخدم ======
    prompt = st.chat_input("Ask me anything...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer the user's question.
            If the answer is not in the context, just say you don't know. 
            Do not invent information. 
            Write the answer in clear, natural language formatted as a short, well-structured article.
            Avoid using any special tokens, symbols, or placeholders.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",  # free, Groq-hosted model
                    temperature=0.0,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_documents = response["source_documents"]

            # 🔹 تجميع الصفحات حسب الملف
            grouped_sources = {}
            for doc in source_documents:
                source = doc.metadata.get("source", "Unknown")
                page = doc.metadata.get("page", "?")
                grouped_sources.setdefault(source, []).append(page)

            # 🔹 تنسيق عرض المصادر
            formatted_sources = "\n".join(
                [f"- **{src}** (pages: {', '.join(map(str, sorted(set(pages))))})"
                 for src, pages in grouped_sources.items()]
            )

            result_to_show = f"{result}\n\n**Source Docs:**\n{formatted_sources}"

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
