from backend import (
    initialise_llm,
    preprocess_data,
    initialise_RAG,
    create_retrival_chain,
    get_answer,
)
import streamlit as st

# Initialise session states
session_states = {
    "vectorstore_initialised": False,
    "vectorstore": None,
}

for session_state, initial_value in session_states.items():
    if session_state not in st.session_state:
        st.session_state[session_state] = initial_value

# Start backend
llm = initialise_llm()

# Initialise vectorstore only once
if st.session_state["vectorstore_initialised"] == False:
    st.session_state["vectorstore"] = preprocess_data()
    st.session_state["vectorstore_initialised"] = True

# Create frontend
st.title("🔗 Pre Screening App")

reset = st.button("Reset")
if reset:
    if st.session_state["vectorstore"] is not None:
        st.session_state["vectorstore"].delete_collection()
    st.session_state["vectorstore_initialised"] = False
    st.rerun()

# Input form
with st.form("my_form"):
    user_input = st.text_area(
        "Enter a reference:",
        placeholder="Yalcin, G., Lim, S., Puntoni, S., & van Osselaer, S. M. J. (2022). Thumbs Up or Down: Consumer Reactions to Decisions by Algorithms Versus Humans",
    )

    # Submit button
    submitted = st.form_submit_button("Submit")
    if submitted & (user_input == ""):
        st.warning("Please enter a question", icon="⚠")
        st.stop()

    # Output field
    if submitted:
        query_transformer = initialise_RAG(
            vectorstore=st.session_state["vectorstore"], llm=llm, query=user_input
        )
        retrival_chain = create_retrival_chain(
            llm=llm, query_transformer=query_transformer
        )

        # Get answer
        answer = get_answer(query=user_input, query_chain=retrival_chain)
        st.info(answer)
