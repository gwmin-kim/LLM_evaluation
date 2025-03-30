from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer
from typing import List
import torch


class MyBGEM3EmbeddingModel(Embeddings):
    def __init__(self, model: str, embed_batch_size: int = 2):
        self.model = AutoModel.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.embed_batch_size = embed_batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state[:,0]
        norm_last_hidden_states = torch.nn.functional.normalize(last_hidden_states, p=2.0, dim=1)
        embeddings = norm_last_hidden_states.tolist()[0]

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])
