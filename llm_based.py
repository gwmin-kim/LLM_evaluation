import os
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter


if __name__ == "__main__":
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    loader = DirectoryLoader(os.getenv("FILE_DIR"), glob="SPRI_AI_Brief_2023년12월호_F.pdf")
    docs = loader.load()

    docs = docs[3:-1]
    for doc in docs:
        doc.metadata["filename"] = doc.metadata["source"]

    generator_llm = LangchainLLMWrapper(llm)
    generator_embeddings = LangchainEmbeddingsWrapper(embeddings=embeddings)

    # persist_db = Chroma(
    #     persist_directory=os.getenv("DB_PATH"),
    #     collection_name=os.getenv("DB_COLLECTION_NAME"),
    #     embedding_function=embeddings
    # )

    # print(persist_db.get())

    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)
    dataset = generator.generate_with_langchain_docs(docs, testset_size=10)