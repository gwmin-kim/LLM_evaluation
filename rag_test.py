
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from clients import PDFRAG
# from models import MyBGEM3EmbeddingModel


if __name__ == "__main__":
    load_dotenv()
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    persist_db = Chroma(
        persist_directory=os.getenv("DB_PATH"),
        collection_name=os.getenv("DB_COLLECTION_NAME"),
        embedding_function=embeddings
    )

    agent = PDFRAG(vectorstore=persist_db, llm=llm, embedding=embeddings)

    q1 = "기업별 LLM 또는 생성형 AI 모델 이름을 알려줘"
    q2 = "알리바바의 최신 LLM 모델 이름 알려줘"

    for q in [q1, q2]:
        chain = agent.create_chain(3, 0.3)
        contexts = agent.get_similar_docs(q, 3)
        print("## This is context ##\n", contexts[0].page_content)
        response = chain.invoke(q)
        print(response)