from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains import StuffDocumentsChain, LLMChain
from handlers import TracingCallbackHandler


dotenv_status = load_dotenv()
logger.info(f"Dotenv loaded: {dotenv_status}")

def ask_llm(question: str) -> str:
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a world class technical documentation writer."),
        ("user", "{input}")
    ])
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    response = chain.invoke({"input": question})

    return response

def ask_rag(question: str) -> str:
    # LLM and Embeddings
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    
    # Docs to VS
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Horned_sungem")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    # Template
    template = """Answer the question based only on the following context:

    {context}

    Question: {input}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Parser
    # output_parser = StrOutputParser()

    # LCEL chain
    # chain = {"context": retriever | documents, "question": RunnablePassthrough()} | prompt | llm | output_parser
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    # chain = retrieval_chain | output_parser

    response = retrieval_chain.invoke({"input": question})

    return response['answer']


def ask_broken_rag(question: str) -> str:
    # LLM and Embeddings
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    
    # Docs to VS
    loader = WebBaseLoader("https://en.wikipedia.org/wiki/Horned_sungem")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)
    vector = FAISS.from_documents(documents, embeddings)
    retriever = vector.as_retriever()

    # Template
    template = """Answer the question based only on the following context:

    {context}

    Question: {input}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Parser
    output_parser = StrOutputParser()

    # LCEL chain
    # chain = {"context": retriever | documents, "question": RunnablePassthrough()} | prompt | llm | output_parser
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    chain = retrieval_chain | output_parser

    response = chain.invoke({"input": question})

    return response


def ask_legacy_rag(question: str) -> str:
    from opentelemetry.instrumentation.langchain import LangchainInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    LangchainInstrumentor().uninstrument()
    tracer = TracerProvider()
    
    # Tracing
    tracing_handler = TracingCallbackHandler(tracer)

    # LLM and Embeddings
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", callbacks=[tracing_handler])
    
    # Docs to VS
    # loader = WebBaseLoader("https://en.wikipedia.org/wiki/Horned_sungem")
    # docs = loader.load()
    # text_splitter = RecursiveCharacterTextSplitter()
    # documents = text_splitter.split_documents(docs)
    # vector = FAISS.from_documents(documents, embeddings)
    # retriever = vector.as_retriever()

    # Template
    prompt = PromptTemplate.from_template(
        "Answer this question: {context}"
    )

    # Parser
    # output_parser = StrOutputParser()

    # LCEL chain
    # chain = {"context": retriever | documents, "question": RunnablePassthrough()} | prompt | llm | output_parser
    # document_chain = create_stuff_documents_chain(llm, prompt)
    # retrieval_chain = create_retrieval_chain(retriever, document_chain)
    # chain = retrieval_chain | output_parser

    llm_chain = LLMChain(llm=llm, prompt=prompt, callbacks=[tracing_handler])
    # chain = StuffDocumentsChain(
    #     llm_chain=llm_chain,
    #     document_prompt=document_prompt,
    #     document_variable_name=document_variable_name,
    #     callbacks=[tracing_handler]
    # )

    response = llm_chain.invoke(question)

    return response['text']