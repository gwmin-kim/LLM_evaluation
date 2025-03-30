
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

from clients import PDFRAG
# from models import MyBGEM3EmbeddingModel

from ragas.testset.graph import Node, KnowledgeGraph



if __name__ == "__main__":
    load_dotenv()
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    persist_db = Chroma(
        persist_directory=os.getenv("DB_PATH"),
        collection_name=os.getenv("DB_COLLECTION_NAME"),
        embedding_function=embeddings
    )
    retriever = persist_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.3})

    results = retriever.invoke("기업별 LLM 또는 생성형 AI 모델 이름을 알려줘")
    # results = retriever.invoke("알리바바의 최신 LLM 모델 이름 알려줘")
    for result in results:
        print(result.page_content)

 