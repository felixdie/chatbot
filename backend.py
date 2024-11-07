from config.ingest_config import config
import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
import streamlit as st
import logging
import re

##################################### Logger ####################################################


def get_logger() -> logging.Logger:
    """
    Creates a logger to debug in console.

    Parameters:
        None.

    Returns:
        logger: (logging.Logger): Configured logger instance that outputs messages to the console.
    """
    # Set up logger
    logger = logging.getLogger(__name__)
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = get_logger()


##################################### Modules ####################################################


def initialise_llm() -> RunnablePassthrough:
    """
    Initialise the LLM model with the necessary configurations.

    Parameters:
        None
    Returns:
        llm (ChatOpenAI): The configured LLM model.
    """
    # Load API key
    dotenv.load_dotenv(dotenv_path="api_key/.env")
    api_key = os.getenv("OPENAI_API_KEY")

    # Configure model
    llm = ChatOpenAI(
        model=config["backend"]["llm_model"],
        temperature=config["backend"]["llm_temparature"],
        api_key=api_key,
        max_retries=config["backend"]["max_retries"],
    )

    return llm


def preprocess_data(
    task_1: bool,
    task_1_1: bool,
    task_2: bool,
    user_input: str,
    llm: RunnablePassthrough,
) -> Chroma:
    """
    Initialise the RAG model with the necessary configurations.

    Parameters:
        task_1 (bool): Whether the user ticked task 1.
        task_1_1 (bool): Whether the user ticked task 1.1.
        task_2 (bool): Whether the user ticked task 2.
    Returns:
        vectorstore (Chroma): The vector store containing the document chunks.
    """
    # Initialise Preprocessing Agent
    agent = Preprocessing_Agent()

    if task_1:

        # Let Preprocessing Agent construct link to data
        filepath = agent.get_filepath(user_input=user_input, llm=llm)
        logger.info(f"SUCCESS: Link constructed {filepath}")

        # Initialise document loader to pull text from web
        try:
            loader = WebBaseLoader(filepath)
            logger.info(f"SUCCESS: Data fetched")

        except:
            logger.info(f"ERROR: Data not fetched")
            st.warning(
                "Paper could not be retrieved from backend. Please try another reference."
            )
            st.stop()

    elif task_1_1:
        loader = WebBaseLoader(config["backend"]["data_task_1_1"])

    elif task_2:
        loader = WebBaseLoader(config["backend"]["data_task_2"])

    # Load data
    data = loader.load()
    logger.info("Data loaded successfully")

    # Initialise text splitter
    if task_1:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["backend"]["chunk_size_task_1"],
            chunk_overlap=config["backend"]["chunk_overlap_task_1"],
            add_start_index=True,
        )

    elif task_1_1:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["backend"]["chunk_size_task_1_1"],
            chunk_overlap=config["backend"]["chunk_overlap_task_1_1"],
            add_start_index=True,
        )

    elif task_2:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["backend"]["chunk_size_task_2"],
            chunk_overlap=config["backend"]["chunk_overlap_task_2"],
            add_start_index=True,
        )

    # Split data into chunks
    all_chunks = text_splitter.split_documents(data)
    logger.info(f"SUCCESS: {len(all_chunks)} chunks extracted")

    # If tasks are switched, clear vectorstore with data from Task 1 (stored by default)
    try:
        if vectorstore._collection.count() > 0:
            vectorstore.delete_collection()

    except:
        # Embed and store chunks in vector store
        vectorstore = Chroma.from_documents(
            documents=all_chunks, embedding=OpenAIEmbeddings()
        )

        # Status logging for vectorstore
        if len(all_chunks) > vectorstore._collection.count():
            logger.info(
                f"ERROR: Vectorstore storage exceeded. {len(all_chunks)-vectorstore._collection.count()} chunks not uploaded to vectorstore"
            )

        elif len(all_chunks) == vectorstore._collection.count():
            logger.info(
                f"SUCCESS: All extracted chunks uploaded to vectorstore ({vectorstore._collection.count()})"
            )

        elif len(all_chunks) < vectorstore._collection.count():
            logger.info(
                f"ERROR: {vectorstore._collection.count()-len(all_chunks)} old chunks in vectorstore, clear by clicking Reset"
            )

        return vectorstore


class Preprocessing_Agent:
    """
    A class representing an Agent that preprocesses data for the RAG model.

    Methods:
        get_filepath: Constructs a link to the paper based on the name that the user provides.
    """

    def __init__(self):
        pass

    def get_filepath(self, user_input: str, llm: RunnablePassthrough) -> str:
        """
        Constructs a link to the paper based on the name that the user provides.

        Parameters:
            user_input (str): The user's input.
            llm (RunnablePassthrough): The LLM model.

        Returns:
            filepath (str): The link to the paper.

        """
        # Initialise preprocessing Agent for task 1
        prompt = """
        You are a helpful assistant. Your task is to extract the title of the scientific paper from the reference provided by the user. Only return the title of the paper.
        """

        question = user_input

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ]

        # Pass user input to LLM
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ]

        # LLM returns title from user input
        response = llm.invoke(messages)
        paper_title = response.content
        logger.info(f"SUCCESS: Paper title extracted {paper_title}")

        # Convert paper title to lowercase letters and replace blanks with underscores
        paper_title = paper_title.lower().replace(" ", "_")

        # Replace characters resulting in invalid filenames with underscores
        paper_title = re.sub(r'[,<>=:"/\\|?*+.%#&{}@`´\']', "", paper_title)

        # Construct link to data
        filepath = (
            "https://raw.githubusercontent.com/felixdie/chatbot/refs/heads/main/data/"
            + paper_title
            + ".txt"
        )

        return filepath


def initialise_RAG(
    query: str,
    vectorstore: Chroma,
    llm: ChatOpenAI,
    task_1: bool,
    task_1_1: bool,
    task_2: bool,
) -> RunnablePassthrough:
    """
    Initialise the RAG model based on the provided papers.

    Parameters:
        query (str): The user's question.
        vectorstore (Chroma): The vector store containing the document chunks.
        llm (ChatOpenAI): The LLM model.

    Returns:
        query_transforming_retriever_chain (RunnablePassthrough): The chain that transforms
            the user's query and retrieves the answer.

    """
    # Initialise retriever with k chunks from vectorstore
    if task_1:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config["backend"]["number_chunks_task_1"]},
        )
    elif task_1_1:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config["backend"]["number_chunks_task_1_1"]},
        )
    elif task_2:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config["backend"]["number_chunks_task_2"]},
        )

    # Retrieve k chunks from vectorstore as context for answer
    # retrieved_docs = retriever.invoke(query)
    # print(f"Chunk 1: {retrieved_docs[0].page_content}\n")
    # print(f"Chunk 2: {retrieved_docs[1].page_content}\n")
    # print(f"Chunk 3: {retrieved_docs[2].page_content}\n")
    # print(f"Chunk 4: {retrieved_docs[3].page_content}\n")
    # print(f"Chunk 5: {retrieved_docs[4].page_content}\n")
    # print(f"Chunk 6: {retrieved_docs[5].page_content}\n")

    # Consider chat history when retrieving chunks
    query_transform_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
            ),
        ]
    )

    # Pass query to retriever
    query_transforming_retriever_chain = RunnableBranch(
        (
            lambda x: len(x.get("messages", [])) == 1,
            # If single message pass directly to retriever
            (lambda x: x["messages"][-1].content) | retriever,
        ),
        # If multiple messages transform query based on chat history before passing to retriever
        query_transform_prompt | llm | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")

    return query_transforming_retriever_chain


def create_retrival_chain(
    llm: ChatOpenAI,
    query_transformer: RunnablePassthrough,
    task_1: bool,
    task_1_1: bool,
    task_2: bool,
) -> RunnablePassthrough:
    """
    Create the chain that retrieves the answer to the user's question.

    Parameters:
        llm (ChatOpenAI): The LLM model.

    Returns:
        conversational_retrieval_chain (RunnablePassthrough): The chain that retrieves
            the answer to the user's question
    """
    # Set system prompt and context for answers
    if task_1:
        SYSTEM_TEMPLATE = config["backend"]["system_prompt_task_1"]
    elif task_1_1:
        SYSTEM_TEMPLATE = config["backend"]["system_prompt_task_1_1"]
    elif task_2:
        SYSTEM_TEMPLATE = config["backend"]["system_prompt_task_2"]

    # Consider context when answering questions
    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                SYSTEM_TEMPLATE,
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    # Fill document chain with context
    document_chain = create_stuff_documents_chain(llm, question_answering_prompt)

    # Pass chat history to retriever to generate query that retrieves context for answer
    conversational_retrieval_chain = RunnablePassthrough.assign(
        context=query_transformer,
    ).assign(
        answer=document_chain,
    )

    return conversational_retrieval_chain


def get_answer(query: str, query_chain: RunnablePassthrough) -> str:
    """
    Get the answer to the user's question based on the query_chain.

    Parameters:
        query (str): The user's question.
        query_chain (RunnablePassthrough): The query chain to retrieve the answer.
    Returns:
        answer (str): The answer to the user's question.
    """
    # Query with context
    response = query_chain.invoke(
        {
            "messages": [
                HumanMessage(content=query),
            ]
        }
    )

    # Return context
    counter = 1
    for chunk in response["context"]:
        print(f"Chunk {counter}: {chunk}\n")
        counter += 1

    return response["answer"]
