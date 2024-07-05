import pandas as pd
import chromadb
from chromadb.config import Settings
from langchain.schema import Document
from langchain.vectorstores import Chroma


class ChromaDB:
    def __init__(self, host, **kwargs):
        port = kwargs.get("port", None)
        if port:
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings(allow_reset=True)
            )
        else:
            self.client = chromadb.PersistentClient(path=host)

    def reset(self):
        for collection in self.client.list_collections():
            self.client.delete_collection(collection.name)
        assert len(self.client.list_collections()) == 0

    def delete_collection(self, collection_name):
        collection_names = [collection.name for collection in self.client.list_collections()]
        if collection_name in collection_names:
            self.client.delete_collection(collection_name)

    def build_collection(self, collection_name, docs, encoder):
        self.delete_collection(collection_name)
        store = Chroma.from_documents([
            Document(page_content=page["page_content"], metadata=page["metadata"]) if page.get("metadata")
            else Document(page_content=page["page_content"])
            for page in docs],
            ids=[page["ids"] for page in docs],
            embedding=encoder,
            collection_name=collection_name,
            client=self.client,
            collection_metadata={"hnsw:space": "cosine"},
            relevance_score_fn=lambda distance: 1.0 - distance / 2.0
            )
        return store

    def get_collection(self, collection_name, encoder):
        collection_names = [collection.name for collection in self.client.list_collections()]
        assert collection_name in collection_names
        store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=encoder,
            collection_metadata={"hnsw:space": "cosine"},
            relevance_score_fn=lambda distance: 1.0 - distance / 2.0
        )
        return store