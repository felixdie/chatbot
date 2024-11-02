import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

from helper_functions.helpers_llm import parse_retriever_input


# Load API key
dotenv.load_dotenv(dotenv_path="config/.env")
api_key = os.getenv("OPENAI_API_KEY")

# Configure model
chat = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key, max_retries=3)

# Initialise document loader to pull text from web
# Change this to pull text from PDFs stored on cloud
loader = WebBaseLoader("https://docs.smith.langchain.com/overview")
data = loader.load()

# Split pulled text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Embed and store chunks in vector store
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# Retrieve k chunks from vectorstore as context for answer
retriever = vectorstore.as_retriever(k=4)

# Set system prompt and context for answers
SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. 
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":

<context>
{context}
</context>
"""

# Set context and placeholder for chat history
question_answering_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_TEMPLATE,
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Provide LLM with context and chat history
document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

# Fill document chain with retrieved info
retrieval_chain = RunnablePassthrough.assign(
    context=parse_retriever_input | retriever,
).assign(
    answer=document_chain,
)

# Ask question with context
response = retrieval_chain.invoke(
    {
        "messages": [
            HumanMessage(content="Can LangSmith help test my LLM applications?")
        ],
    }
)

print(response)
