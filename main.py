from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.llms import HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain import PromptTemplate
import os


def extract_answer(response):
    if "Answer:" in response:
        response = response.split("Answer:")[-1]
    return response.strip()


class ChatBot():
    load_dotenv()
    loader = TextLoader('./horoscope.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=4)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    vectorstore = Chroma.from_documents(
        documents=docs, embedding=HuggingFaceEmbeddings())

    repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    llm = HuggingFaceHub(
        repo_id=repo_id, model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50}, huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
    )

    template = """
  You are a seer. These Human will ask you a questions about their life. Use following piece of context to answer the question. 
  If you don't know the answer, just say I don't know antyhing about this. 
  You answer with short and concise answer, no longer than2 sentences.

  Context: {context}
  Question: {question}
  Answer: 

  """

    prompt = PromptTemplate(template=template, input_variables=[
                            "context", "question"])

    rag_chain = (
        {"context": vectorstore.as_retriever(),  "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        | (lambda res: extract_answer(res))
    )
