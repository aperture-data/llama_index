import os
import uuid
import pytest

from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.ApertureDB import ApertureDBVectorStore

import aperturedb


class MockConnector:
    hello_version = 2

    def __init__(self, exists) -> None:
        self.exists = exists
        self.queries = []

    def clone(self):
        return self

    def query(self, *args, **kwargs):
        print("query called with args:", args, "and kwargs:", kwargs)
        self.queries.append(args[0])
        command = next(iter(args[0][0]))
        result = {"status": 0}
        if command == "GetIndexes":
            result["indexes"] = (
                [
                    {
                        "target": "descriptor",
                        "class": "llamaindex",
                        "property": "_blob",
                        "kind": "hnsw",
                        "params": {"metric": "CS", "dimensions": 1024},
                    }
                ]
                if self.exists
                else []
            )
        return [{command: result}], []

    def last_query_ok(self):
        return True

    def get_last_response_str(self):
        return "response"


def test_class():
    names_of_base_classes = [b.__name__ for b in ApertureDBVectorStore.__mro__]
    assert BasePydanticVectorStore.__name__ in names_of_base_classes


def test_vector_store_initialize_noop_when_DS_exists(monkeypatch):
    monkeypatch.setattr(
        aperturedb.CommonLibrary,
        "create_connector",
        lambda *args, **kwargs: MockConnector(True),
    )
    store = ApertureDBVectorStore(dimensions=1024)
    assert store._dimensions == 1024
    expected_queries = ["GetStatus", "GetIndexes"]
    print(store.client.queries)
    for i, query in enumerate(expected_queries):
        assert query in store.client.queries[i][0]


def test_vector_store_initialize_noop_when_DS_does_not_exist(monkeypatch):
    monkeypatch.setattr(
        aperturedb.CommonLibrary,
        "create_connector",
        lambda *args, **kwargs: MockConnector(False),
    )
    store = ApertureDBVectorStore(dimensions=1024)
    assert store._dimensions == 1024
    expected_queries = [
        "GetStatus",
        "GetIndexes",
        "CreateIndex",
        "GetIndexes",
        "CreateIndex",
    ]
    for i, query in enumerate(expected_queries):
        assert query in store.client.queries[i][0]


def test_vector_store_delete_confines_queries_to_ref_doc_id(monkeypatch):
    monkeypatch.setattr(
        aperturedb.CommonLibrary,
        "create_connector",
        lambda *args, **kwargs: MockConnector(True),
    )
    store = ApertureDBVectorStore(dimensions=1024)
    store.delete("doc_id")
    expected_queries = ["GetStatus", "GetIndexes", "DeleteDescriptor"]
    for i, query in enumerate(expected_queries):
        assert query in store.client.queries[i][0]
        if query == "DeleteDescriptor":
            constraints = store.client.queries[i][0]["DeleteDescriptor"]["constraints"]
            assert constraints == ["$lm_ref_doc_id", "==", "doc_id"]
            assert (
                store.client.queries[i][0]["DeleteDescriptor"]["with_class"]
                == "llamaindex"
            )


def test_vector_store_clear_deletes_all_descriptors(monkeypatch):
    monkeypatch.setattr(
        aperturedb.CommonLibrary,
        "create_connector",
        lambda *args, **kwargs: MockConnector(True),
    )
    store = ApertureDBVectorStore(dimensions=1024)
    store.clear()
    expected_queries = ["GetStatus", "GetIndexes", "DeleteDescriptor"]
    for i, query in enumerate(expected_queries):
        assert query in store.client.queries[i][0]
        if query == "DeleteDescriptor":
            assert "constraints" not in store.client.queries[i][0]["DeleteDescriptor"]
            assert (
                store.client.queries[i][0]["DeleteDescriptor"]["with_class"]
                == "llamaindex"
            )


@pytest.mark.skipif(
    not os.getenv("APERTUREDB_LLAMA_LIVE"),
    reason="Requires a disposable ApertureDB 1.0 instance",
)
def test_native_vector_store_lifecycle():
    from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
    from llama_index.core.vector_stores.types import (
        MetadataFilter,
        MetadataFilters,
        VectorStoreQuery,
    )

    name = "LlamaVector" + uuid.uuid4().hex
    store = ApertureDBVectorStore(
        descriptor_set=name,
        dimensions=2,
        engine="Flat",
        properties={"model": "fixture"},
    )
    try:
        reopened = ApertureDBVectorStore(descriptor_set=name)
        assert reopened._dimensions == 2
        assert reopened._properties == {"model": "fixture"}
        assert name in store.get_descriptor_set()

        def node(id_, text, embedding, group, doc):
            return TextNode(
                id_=id_,
                text=text,
                embedding=embedding,
                metadata={"group": group},
                relationships={NodeRelationship.SOURCE: RelatedNodeInfo(node_id=doc)},
            )

        nodes = [
            node("a", "first", [1.0, 0.0], "$a", "$doc1"),
            node("b", "second", [0.0, 1.0], "b", "doc2"),
        ]
        assert store.add(nodes) == ["a", "b"]
        store.add(nodes[:1])
        q = VectorStoreQuery(query_embedding=[1.0, 0.0], similarity_top_k=3)
        assert store.query(q).ids == ["a", "b"]
        q.filters = MetadataFilters(filters=[MetadataFilter(key="group", value="$a")])
        filtered = store.query(q)
        assert filtered.ids == ["a"]
        assert filtered.nodes[0].text == "first"
        store.delete("$doc1")
        q.filters = None
        assert store.query(q).ids == ["b"]
        store.clear()
        assert store.query(q).ids == []
    finally:
        store.delete_vector_store(name)
