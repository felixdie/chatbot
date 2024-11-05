from config.ingest_config import config
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
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


def preprocess_data(task_1: bool, task_1_1: bool, task_2: bool) -> Chroma:
    """
    Initialise the RAG model with the necessary configurations.

    Parameters:
        task_1 (bool): Whether the user ticked task 1.
        task_1_1 (bool): Whether the user ticked task 1.1.
        task_2 (bool): Whether the user ticked task 2.
    Returns:
        vectorstore (Chroma): The vector store containing the document chunks.
    """
    # Initialise document loader to pull text from web
    if task_1:
        loader = WebBaseLoader(config["backend"]["data_task_1"])
    elif task_1_1:
        loader = WebBaseLoader(config["backend"]["data_task_1_1"])
    elif task_2:
        loader = WebBaseLoader(config["backend"]["data_task_2"])

    data = loader.load()

    # Split pulled text into chunks
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

    all_chunks = text_splitter.split_documents(data)
    print(f"Extracted Chunks: {len(all_chunks)}")

    # When this hits a limit, add chunks in batches
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
            print(
                "Vectorstore storage exceeded: Not all chunks uploaded to vectorstore"
            )
        elif len(all_chunks) == vectorstore._collection.count():
            print("Status OK: All chunks uploaded to vectorstore")
        elif len(all_chunks) < vectorstore._collection.count():
            print("Clear vectorstore: Old chunks are in vectorstore, click Reset")

        return vectorstore


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
    retrieved_docs = retriever.invoke(query)
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

    return response["answer"]

    # Return context
    # for chunk in response_1["context"]:
    #     print(document)
    #     print()
