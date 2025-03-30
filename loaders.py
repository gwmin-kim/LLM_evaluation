from clients import DataProprecessor
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


if __name__ == "__main__":
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    file_path = os.getenv("FILE_PATH")
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    # embedding = MyBGEM3EmbeddingModel(os.getenv("BGE_MODEL_PATH"))
    
    Preprocessor = DataProprecessor(file_path=file_path, embedding=embedding)
    docs = Preprocessor.load_documents()

    ## Pre-processing
    docs = docs[3:-1]
    for doc in docs:
        doc.metadata["filename"] = doc.metadata["source"]
    
    chunks = Preprocessor.split_documents(docs=docs, chunk_size=500, chunk_overlap=50)
    vectorstore = Preprocessor.create_chroma_vectorstore(chunks)