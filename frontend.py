from backend import (
    initialise_llm,
    preprocess_data,
    initialise_RAG,
    create_retrival_chain,
    get_answer,
)
import streamlit as st
from config.ingest_config import config


# Initialise session states
session_states = {
    "vectorstore_initialised": False,
    "vectorstore": None,
}

for session_state, initial_value in session_states.items():
    if session_state not in st.session_state:
        st.session_state[session_state] = initial_value

# Initialise LLM
llm = initialise_llm()

# Create frontend
st.title("🔗 Pre Screening App")

# Reset and task checkboxes
col1, col2, col3, col4 = st.columns(4)

with col1:
    task_1 = st.checkbox("Task 1", value=True)

with col2:
    task_1_1 = st.checkbox("Task 1.1")

with col3:
    task_2 = st.checkbox("Task 2")

with col4:
    reset = st.button("Reset")
    if reset:
        if st.session_state["vectorstore"] is not None:
            st.session_state["vectorstore"].delete_collection()
        st.session_state["vectorstore_initialised"] = False
        st.rerun()

# Input form
with st.form("my_form"):
    if task_1:
        user_input = st.text_area(
            "Enter a reference:",
            placeholder="Yalcin, G., Lim, S., Puntoni, S., & van Osselaer, S. M. J. (2022). Thumbs Up or Down: Consumer Reactions to Decisions by Algorithms Versus Humans",
        )

    elif task_1_1:
        user_input = st.text_area(
            "Enter references to count their methods:",
            placeholder="Voice‐based AI in call center customer service: A natural field experiment.\nEliciting human judgment for prediction algorithms.\n...",
        )

    elif task_2:
        user_input = st.text_area(
            "Enter your question:",
            placeholder="Do a literature review",
        )

    # Submit button
    submitted = st.form_submit_button("Submit")
    if submitted & (user_input == ""):
        st.warning("Please enter a question", icon="⚠")
        st.stop()

    # Output field
    if submitted:

        # Initialise vectorstore only once
        if st.session_state["vectorstore_initialised"] == False:
            st.session_state["vectorstore"] = preprocess_data(
                task_1=task_1,
                task_1_1=task_1_1,
                task_2=task_2,
                user_input=user_input,
                llm=llm,
            )
            st.session_state["vectorstore_initialised"] = True

        query_transformer = initialise_RAG(
            vectorstore=st.session_state["vectorstore"],
            llm=llm,
            query=user_input,
            task_1=task_1,
            task_1_1=task_1_1,
            task_2=task_2,
        )
        retrival_chain = create_retrival_chain(
            llm=llm,
            query_transformer=query_transformer,
            task_1=task_1,
            task_1_1=task_1_1,
            task_2=task_2,
        )

        # Get answer
        answer = get_answer(query=user_input, query_chain=retrival_chain)
        st.info(answer)
