from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS, Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough

class DataProprecessor:
    def __init__(self, file_path: str, embedding=OpenAIEmbeddings(model="text-embedding-3-small")):
        self.file_path = file_path
        self.embedding = embedding
    
    def load_documents(self):
        # 문서 로드(Load Documents)
        loader = PyMuPDFLoader(self.file_path)
        docs = loader.load()
        return docs

    def split_documents(self, docs, chunk_size=100, chunk_overlap=50):
        # 문서 분할(Split Documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        split_documents = text_splitter.split_documents(docs)
        return split_documents

    def create_faiss_vectorstore(self, split_documents, persist_directory="./faiss_db", collection_name="my_db"):
        embeddings = self.embedding
        vectorstore = FAISS.from_documents(
            documents=split_documents, embedding=embeddings, persist_directory=persist_directory, collection_name=collection_name
        )
        return vectorstore
    
    def create_chroma_vectorstore(self, split_documents, persist_directory="./chroma_db", collection_name="my_db"):
        embeddings = self.embedding
        vectorstore = Chroma.from_documents(
            documents=split_documents, embedding=embeddings, persist_directory=persist_directory, collection_name=collection_name
        )
        return vectorstore


class PDFRAG:
    def __init__(self, vectorstore, llm, embedding=OpenAIEmbeddings(model="text-embedding-3-small")):
        self.vectorstore = vectorstore
        self.embedding = embedding
        self.llm = llm

    def get_similar_docs(self, query, k=3):
        return self.vectorstore.similarity_search(query, k)

    def retriever(self, k=3, score_threshold=0.3):
        return self.vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": k, "score_threshold": score_threshold})

    def format_docs(self, docs):
        return '\n'.join(doc.page_content for doc in docs)

    def create_chain(self, k=3, score_threshold=0.3):
        # 프롬프트 생성(Create Prompt)
        
        prompt = PromptTemplate.from_template(
            """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 

        #Context: 
        {context}

        #Question:
        {question}

        #Answer:"""
        )

        # 체인(Chain) 생성
        chain = (
            {
                "context": self.retriever(k=k, score_threshold=score_threshold) | self.format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain