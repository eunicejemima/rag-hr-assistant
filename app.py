import streamlit as st

from dataprocessor import process_documents
from generator import API_KEY, generate_answer
from retriever import Retriever


# Load the retriever once so the embedding model is not reloaded for every question.
@st.cache_resource
def load_retriever():
    retriever = Retriever()

    try:
        retriever.load_vector_store()
    except FileNotFoundError:
        chunks = process_documents()
        retriever.create_vector_store(chunks)

    return retriever


# Turn retrieved chunks into the context that the answer generator expects.
def build_context(results):
    return "\n\n".join(
        f"Source: {result['source']}\n{result['text']}"
        for result in results
    )


# Set up the page and explain what the assistant does.
st.set_page_config(
    page_title="RAG HR Assistant",
    page_icon=":briefcase:",
)

st.title("RAG HR Assistant")
st.write("Ask a question about the company HR policies.")


# Show a clear message instead of allowing a missing API key to crash the app.
if not API_KEY:
    st.error(
        "GEMINI_API_KEY is missing. Add it to your .env file, then restart Streamlit."
    )
    st.stop()


# Keep all earlier questions and answers for this browser session.
if "messages" not in st.session_state:
    st.session_state.messages = []


# These buttons place a useful starter question into the chat.
st.subheader("Try a sample question")
sample_columns = st.columns(3)
sample_questions = {
    "Sick leave": "What is the policy for sick leave?",
    "Work from home": "What is the work-from-home policy?",
    "Laptop lost": "What should I do if I lose my company laptop?",
}

for column, (label, question) in zip(sample_columns, sample_questions.items()):
    if column.button(label, use_container_width=True):
        st.session_state.pending_question = question


# Display the conversation saved in this session.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            st.caption(f"Sources: {', '.join(message['sources'])}")


# Accept either a typed question or one selected with a sample button.
question = st.chat_input("Ask an HR question...")
if question is None:
    question = st.session_state.pop("pending_question", None)


# Search the policy documents and generate an answer for the new question.
if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )
    with st.chat_message("user"):
        st.markdown(question)

    try:
        retriever = load_retriever()
        with st.chat_message("assistant"):
            with st.spinner("Searching the HR documents and generating an answer..."):
                results = retriever.search(question, top_k=5)
                context = build_context(results)
                answer = generate_answer(question, context)
                sources = list(dict.fromkeys(result["source"] for result in results))

            st.markdown(answer)
            if sources:
                st.caption(f"Sources: {', '.join(sources)}")

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources,
            }
        )
    except ValueError as error:
        st.error(str(error))
    except FileNotFoundError as error:
        st.error(f"Could not load the HR documents: {error}")
