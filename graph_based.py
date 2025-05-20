import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
import asyncio

from ragas.testset.graph import Node, KnowledgeGraph
from ragas.testset.transforms.extractors import NERExtractor
from ragas.testset.transforms.relationship_builders.traditional import JaccardSimilarityBuilder
from ragas.testset.transforms import apply_transforms, Parallel
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import default_query_distribution, single_hop, multi_hop


class GraphBased():
    load_dotenv()
    def __init__(self, vector_db=None, file_path=None):
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.file = file_path
        self.vectordb = vector_db
        self.extractor = NERExtractor()
        self.kg = KnowledgeGraph()
        self.rel_builder = JaccardSimilarityBuilder(property_name="entities", new_property_name="entity_jaccard_similarity")
    
    async def _ner_extract(self, nodes):
        output = [await self.extractor.extract(node) for node in nodes]
        return output

    async def _kg_build(self, nodes):
        self.kg = KnowledgeGraph(nodes=nodes)
        relationships = await self.rel_builder.transform(self.kg)
        return relationships
    
    def abuild_relationships(self):
        sample_nodes = []
        for chunk in self.vectordb.get()["documents"]:
            sample_nodes.append(Node(properties={"page_content": chunk}))    

        output = asyncio.run(self._ner_extract(sample_nodes))
        _ = [node.properties.update({key:val}) for (key,val), node in zip(output, sample_nodes)]
        print(len(sample_nodes), sample_nodes[0].properties)

        rels = asyncio.run(self._kg_build(sample_nodes))
        for rel in rels:
            print(rel.source.properties.get("entities", ""))
            # print(rel.source.properties.get("page_content", ""))
            print(rel.target.properties.get("entities", ""))
            # print(rel.target.properties.get("page_content", ""))
            print(rel.properties)
    
    def transform(self):
        transforms = [self.extractor, self.rel_builder]
        # tranforms = [
        #     Parallel(
        #         KeyphraseExtractor(),
        #         NERExtractor()
        #     ),
        #     rel_builder
        # ]

        apply_transforms(self.kg, transforms)
    
    def synthesize_queries(self):
        generator = TestsetGenerator(llm=self.llm, embedding_model=self.embeddings, knowledge_graph=self.kg)
        query_distribution = default_query_distribution(self.llm)
        testset = generator.generate(testset_size=10, query_distribution=query_distribution)
        testset.to_pandas("test.csv")




#from dataclasses import dataclass
# from ragas.testset.synthesizers.base_query import QuerySynthesizer

# @dataclass
# class EntityQuerySynthesizer(QuerySynthesizer):

#     async def _generate_scenarios( self, n, knowledge_graph, callbacks):
#         """
#         logic to query nodes with entity
#         logic describing how to combine nodes,styles,length,persona to form n scenarios
#         """

#         return scenarios

#     async def _generate_sample(
#         self, scenario, callbacks
#     ):

#         """
#         logic on how to use tranform each scenario to EvalSample (Query,Context,Reference)
#         you may create singleturn or multiturn sample
#         """

#         return SingleTurnSample(user_input=query, reference_contexs=contexts, reference=reference)


if __name__ == "__main__":
    load_dotenv()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    persist_db = Chroma(
        persist_directory=os.getenv("DB_PATH"),
        collection_name=os.getenv("DB_COLLECTION_NAME"),
        embedding_function=embeddings
    )

    graph = GraphBased(vector_db=persist_db)
    # result = graph.abuild_relationships()
    graph.transform()
    graph.synthesize_queries()

    # _ner_extractor와 _rel_build를 통해서 rel을 뽑을 수도 있지만, transform을 통해서도 만들 수 있는 듯 하다. 여기서 부터 살펴본면 됨