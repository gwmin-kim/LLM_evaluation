import os
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_chroma import Chroma

class LLMBased:
    load_dotenv()
    def __init__(self, persist_db=None, file_path=None):
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.file = file_path
        self.vectordb = persist_db
    
    def _set_loader(self, file_path):
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["filename"] = doc.metadata["source"]
        self.docs = docs

    def synthesize_qa_set(self, result_path, test_size=50):
        generator_llm = LangchainLLMWrapper(self.llm)
        generator_embeddings = LangchainEmbeddingsWrapper(embeddings=self.embeddings)
        generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

        self._set_loader(self.file)
        dataset = generator.generate_with_langchain_docs(self.docs, testset_size=test_size)
        
        result = dataset.to_pandas()
        result.to_csv(result_path)

if __name__ == "__main__":
    test_class = LLMBased(file_path="./data/HiRAG_2503.10150v1.pdf")
    test_class.synthesize_qa_set("./llm_based_synthetic_qa_set.csv")

