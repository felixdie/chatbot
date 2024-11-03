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

# get query, return answer


def initialise_llm() -> RunnablePassthrough:
    """
    Initialise the LLM model with the necessary configurations.

    Parameters:
        None
    Returns:
        llm (ChatOpenAI): The configured LLM model.
    """
    # Load API key
    dotenv.load_dotenv(dotenv_path="config/.env")
    api_key = os.getenv("OPENAI_API_KEY")

    # Configure model
    llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=0.2, api_key=api_key, max_retries=3
    )

    return llm


def initialise_rag(llm: ChatOpenAI) -> RunnablePassthrough:
    """
    Initialise the RAG model with the necessary configurations.

    Parameters:
        None
    Returns:
        conversational_retrieval_chain (RunnablePassthrough): The chain that retrieves
            the answer to the user's question.
    """
    # Initialise document loader to pull text from web
    loader = WebBaseLoader(
        "https://raw.githubusercontent.com/felixdie/chatbot/refs/heads/main/data/data.txt"
    )

    data = loader.load()

    # Split pulled text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(data)

    # print(len(all_splits))

    # Embed and store chunks in vector store
    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=OpenAIEmbeddings()
    )

    # Retrieve k chunks from vectorstore as context for answer
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 6}
    )

    # Check context e.g. first chunk
    # retrieved_docs = retriever.invoke("Summarise the study")
    # print(retrieved_docs[0].page_content)

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

    # Set system prompt and context for answers
    SYSTEM_TEMPLATE = """
    You are an assistant for question-answering tasks. Answer the user's questions based on the below context.
    If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know".
    Keep the answer as concise as possible.

    <context>
    {context}
    </context>
    """

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
        context=query_transforming_retriever_chain,
    ).assign(
        answer=document_chain,
    )

    return conversational_retrieval_chain


def get_answer(query: str, query_chain: RunnablePassthrough) -> str:
    """
    Get the answer to the user's question.

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

    # # Follow up question
    # # response_2 = conversational_retrieval_chain.invoke(
    # #     {
    # #         "messages": [
    # #             HumanMessage(content="Can LangSmith help test my LLM applications?"),
    # #             AIMessage(
    # #                 content="Yes, LangSmith can help test and evaluate your LLM applications. It allows you to quickly edit examples and add them to datasets to expand the surface area of your evaluation sets or to fine-tune a model for improved quality or reduced costs. Additionally, LangSmith can be used to monitor your application, log all traces, visualize latency and token usage statistics, and troubleshoot specific issues as they arise."
    # #             ),
    # #             HumanMessage(content="Tell me more!"),
    # #         ],
    # #     }
    # # )
    # # print(f"Response 2: {response_2["answer"]}")
