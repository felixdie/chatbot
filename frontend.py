__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from backend import (
    initialise_llm,
    preprocess_data,
    initialise_RAG,
    create_retrival_chain,
    get_answer,
    get_logger,
    Master_Agent,
)
import streamlit as st
import chromadb
from config.ingest_config import config


# Initialise logger
logger = get_logger()


# Initialise session states
session_states = {
    "vectorstore_initialised": False,
    "vectorstore": None,
}
for session_state, initial_value in session_states.items():
    if session_state not in st.session_state:
        st.session_state[session_state] = initial_value

# Create frontend
st.title("🔗 GPT Tool for HAI Literature Review")


# col1, col2, col3, col4 = st.columns(4)
# with col1:

# Reset vectorstore and reload frontend
# reset = st.button("Reset")
# if reset:
#     if st.session_state["vectorstore"] is not None:
#         st.session_state["vectorstore"].delete_collection()
#     st.session_state["vectorstore_initialised"] = False
#     logger.info("SUCCESS: Vectorstore cleared")
#     st.rerun()

# Input form
with st.form("my_form"):
    user_input = st.text_area(
        "Enter your question:\n",
        placeholder="Summarise AI in Business Research by Cao et al. (2024) \nConduct a literature review on Conceptual/ Case Studies about human + AI\n...",
    )

    # Submit button
    submitted = st.form_submit_button("Submit")
    if submitted & (user_input == ""):
        st.warning("Please enter a question", icon="⚠")
        st.stop()

    # Output field
    if submitted:

        # Initialise master agent
        master_agent = Master_Agent()

        # Determine task
        task = master_agent.choose_task(user_input=user_input)

        # Initialise LLM
        llm = initialise_llm(task=task)

        # Initialise vectorstore only once
        if st.session_state["vectorstore_initialised"] == False:
            st.session_state["vectorstore"] = preprocess_data(
                task=task, user_input=user_input
            )
            st.session_state["vectorstore_initialised"] = True

        query_transformer = initialise_RAG(
            vectorstore=st.session_state["vectorstore"],
            llm=llm,
            query=user_input,
            task=task,
        )
        logger.info("SUCCESS: RAG initialised")

        retrival_chain = create_retrival_chain(
            llm=llm, query_transformer=query_transformer, task=task
        )
        logger.info("SUCCESS: Retrival Chain initialised")

        # Get answer
        answer = get_answer(query=user_input, query_chain=retrival_chain)
        logger.info("SUCCESS: Answer generated")
        st.info(answer)

        if st.session_state["vectorstore"] is not None:
            st.session_state["vectorstore"].delete_collection()
            st.session_state["vectorstore_initialised"] = False
            chromadb.api.client.SharedSystemClient.clear_system_cache()
            logger.info("SUCCESS: Vectorstore cleared")
