from backend import (
    initialise_llm,
    preprocess_data,
    initialise_RAG,
    create_retrevial_chain,
    get_answer,
)
import streamlit as st

# Start backend
llm = initialise_llm()
vectorstore = preprocess_data()
query_transformer = initialise_RAG(vectorstore=vectorstore, llm=llm)
retrival_chain = create_retrevial_chain(llm=llm, query_transformer=query_transformer)

# Create frontend
st.title("🔗 Pre Screening App")

# Input form
with st.form("my_form"):
    user_input = st.text_area(
        "Enter your question:",
        "...",
    )

    # Submit button
    submitted = st.form_submit_button("Submit")
    if submitted & ((user_input == "...") or (user_input == "")):
        st.warning("Please enter a question", icon="⚠")
        st.stop()

    # Output field
    if submitted:
        answer = get_answer(query=user_input, query_chain=retrival_chain)
        st.info(answer)
