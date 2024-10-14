from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from pprint import pprint
from langchain_community.vectorstores.faiss import FAISS


load_dotenv()


def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    split_docs = splitter.split_documents(docs)
    pprint(len(split_docs))
    return split_docs


def create_db(documents):
    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embedding=embedding)
    return vector_store


def create_chain(vector_store):
    model = ChatOpenAI(model="gpt-4o", temperature=0.4)

    prompt = ChatPromptTemplate.from_template(""" +
             "Answer the user's question:" +
             "Context: {context}" +
             "Question: {input}" +
             """)

    # chain = prompt | model
    chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    retriever_chain = create_retrieval_chain(
        retriever,
        chain
    )

    return retriever_chain


docs = get_documents_from_web('https://www.kychub.com/blog/transaction-monitoring-process/')
vector_store = create_db(docs)
chain = create_chain(vector_store)
response = chain.invoke({"input": "what are Transaction Monitoring Process?"})
pprint(response)