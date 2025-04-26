import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import asyncio
from clients import PDFRAG

from ragas.testset.graph import Node, KnowledgeGraph
from ragas.testset.transforms.extractors import NERExtractor
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder

async def ner_extract(nodes):
    extractor = NERExtractor()
    output = [await extractor.extract(node) for node in nodes]
    return output


async def kg_build(nodes):
    kg = KnowledgeGraph(nodes=nodes)
    rel_builder = JaccardSimilarityBuilder(property_name="entities", new_property_name="entity_jaccard_similarity")
    relationships = await rel_builder.transform(kg)
    return relationships


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

    sample_nodes = []
    for chunk in persist_db.get()["documents"]:
        sample_nodes.append(Node(properties={"page_content": chunk}))    

    print(len(sample_nodes), sample_nodes[0])

    output = asyncio.run(ner_extract(sample_nodes))
    print("-- NER")
    print(output[0])

    print("-- sample nodes")
    _ = [node.properties.update({key:val}) for (key,val), node in zip(output, sample_nodes)]
    print(sample_nodes[0].properties)

    print("-- kg")
    rel = asyncio.run(kg_build(sample_nodes))
    print(len(rel), rel[0])