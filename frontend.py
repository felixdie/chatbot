from backend import initialise_llm, initialise_rag, get_answer
import streamlit as st

llm = initialise_llm()
query_chain = initialise_rag(model=llm)

st.title("🔗 Pre Screening App")

with st.form("my_form"):
    user_input = st.text_area(
        "Enter your question:",
        "...",
    )
    submitted = st.form_submit_button("Submit")
    if submitted:
        answer = get_answer(query=user_input, query_chain=query_chain)
        st.info(answer)


# drop down with 35 papers -> select the one to analyse -> then load
