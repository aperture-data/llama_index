"""
ApertureDB vector store index.

"""

from __future__ import annotations

import json
import logging
import numpy as np

from pydantic import PrivateAttr
from typing_extensions import override
from llama_index.core.schema import TextNode
from typing import Any, Dict, List, Optional
from llama_index.core.vector_stores.types import (
    BasePydanticVectorStore,
    MetadataFilters,
    VectorStoreQuery,
    VectorStoreQueryResult,
)

from llama_index.core.vector_stores.types import FilterOperator, FilterCondition


from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
)
from llama_index.core.schema import MetadataMode
from aperturedb.ParallelLoader import ParallelLoader
from aperturedb.Constraints import Constraints, Conjunction, predicate
from aperturedb.ServerCompatibility import preferred_vector_index

# Default descriptorset name
DESCRIPTOR_SET = "llamaindex"

## Defaults as defined in the ApertureDB SDK
ENGINE = "HNSW"
METRIC = "CS"

BATCHSIZE = 1000

# Prefix for properties that are in the client metadata
PROPERTY_PREFIX = "lm_"

TEXT_PROPERTY = "text"  # Property name for the text
UNIQUEID_PROPERTY = "uniqueid"  # Property name for the unique id


class ApertureDBVectorStore(BasePydanticVectorStore):
    stores_text: bool = True
    flat_metadata: bool = True
    logger: logging.Logger = logging.getLogger(__name__)

    _client = PrivateAttr()

    """
    ApertureDB vectorstore.

    This VectorStore uses DescriptorSet to store the embeddings and metadata.

    Query is run with FindDescriptor to find k most similar embeddings.

    Args:
        embeddings (Embeddings): Embedding function.

        descriptor_set (str, optional): Descriptor set name. Defaults to "llamaindex".

        dimensions (Optional[int], optional):   Number of dimensions of the embeddings.
            Defaults to None.

        engine (str, optional): Engine to use.
            Defaults to "HNSW" for new descriptorsets.

        metric (str, optional): Metric to use. Defaults to "CS" for new descriptorsets.

        log_level (int, optional): Logging level. Defaults to logging.WARN.
        overwrite (bool, optional): Default set to True. Will overwrite existing descriptor set.
    Example:

        ```python
            # Get the data for running the example
            # mkdir -p 'data/paul_graham/'
            # wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'

            from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
            from llama_index.core import StorageContext
            from llama_index.vector_stores.ApertureDB import ApertureDBVectorStore

            adb_client = ApertureDBVectorStore(dimensions=1536)
            storage_context = StorageContext.from_defaults(vector_store=adb_client)


            documents = SimpleDirectoryReader("./data/paul_graham/").load_data()
            index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

            query_engine = index.as_query_engine()

            def run_queries(query_engine):
                query_str = [
                    "What did the author do growing up?",
                    "What did the author do after his time at Viaweb?"
                ]
                for qs in query_str:
                    response = query_engine.query(qs)
                    print(f"{qs=}\r\n")
                    print(response)

            run_queries(query_engine)

            # Delete the first document from the index.
            # This particular example folder has just 1 document.
            # Deleting it would cause the queries to return empty results.
            index.delete(documents[0].doc_id)

            run_queries(query_engine)
    """

    @override
    def __init__(
        self,
        descriptor_set: str = DESCRIPTOR_SET,
        embeddings: Any = None,
        dimensions: Optional[int] = None,
        engine: Optional[str] = None,
        metric: Optional[str] = None,
        log_level: int = logging.WARN,
        properties: Optional[Dict] = None,
        overwrite: bool = True,
        **kwargs: Any,
    ) -> None:
        # ApertureDB imports
        try:
            from aperturedb.Utils import Utils
            from aperturedb.CommonLibrary import create_connector
            from aperturedb.Descriptors import Descriptors

        except ImportError:
            raise ImportError(
                "Install the ApertureDB SDK with pip install 'aperturedb>=1.0.3'."
            )

        super().__init__(**kwargs)

        self.logger.setLevel(log_level)
        self._descriptor_set = descriptor_set
        self._dimensions = dimensions
        self._engine = engine
        self._metric = metric
        self._properties = properties
        self._overwrite = overwrite
        # TODO: Either standardize this or remove it.
        self._embedding_function = embeddings

        ## Returns a client for the database
        self._client = create_connector()
        self._descriptors = Descriptors(self._client)
        self._utils = Utils(self._client)

        try:
            self._utils.status()
        except Exception:
            self.logger.exception("Failed to connect to ApertureDB")
            raise

        self._find_or_add_descriptor_set()  ## Call to find or add a descriptor set

    @classmethod
    def class_name(cls) -> str:
        return "ApertureDBVectorStore"

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def _find_or_add_descriptor_set(self) -> None:
        """
        Checks if the descriptor set exists, if not, creates it.
        """
        indexes = self._utils.get_vector_indexes(self._descriptor_set)
        if indexes:
            index = preferred_vector_index(indexes, self._engine, self._metric)
            if index is None:
                raise ValueError(
                    "Requested engine or metric is absent from the vector indexes"
                )
            params = index["params"]
            if (
                self._dimensions is not None
                and self._dimensions != params["dimensions"]
            ):
                raise ValueError(
                    "Embedding dimensions do not match the existing vector index"
                )
            self._dimensions = params["dimensions"]
            self._engine = index["kind"].upper()
            metrics = params["metric"]
            self._metric = self._metric or (
                metrics if isinstance(metrics, str) else metrics[0]
            )
            self._properties = {
                key[len(PROPERTY_PREFIX) :]: value
                for key, value in index.get("metadata", {}).items()
                if key.startswith(PROPERTY_PREFIX)
            }
            return

        self._engine = self._engine or ENGINE
        self._metric = self._metric or METRIC
        if self._dimensions is None:
            if self._embedding_function is None:
                raise ValueError("Dimensions or an embedding function must be provided")
            self._dimensions = len(self._embedding_function.get_text_embedding("test"))
        metadata = {PROPERTY_PREFIX + k: v for k, v in (self._properties or {}).items()}
        if not self._utils.create_vector_index(
            self._descriptor_set, self._dimensions, self._metric, self._engine, metadata
        ):
            raise RuntimeError(f"Cannot create vector index for {self._descriptor_set}")
        if not self._utils.create_entity_index(
            f"descriptor:{self._descriptor_set}", UNIQUEID_PROPERTY
        ):
            raise RuntimeError(
                f"Cannot index descriptor IDs for {self._descriptor_set}"
            )

    def add(
        self,
        nodes: List[TextNode],
        **kwargs: Any,
    ) -> List[str]:
        """
        Adds a list of nodes as Descriptors to the Descriptorset.

        Args:
            nodes: List[TextNode] List of text nodes
            kwargs: Additional arguments to pass to add

        """
        ids = []
        data = []

        for node in nodes:
            metadata = node_to_metadata_dict(
                node, remove_text=False, flat_metadata=self.flat_metadata
            )

            properties = {
                UNIQUEID_PROPERTY: node.node_id,
                "text": node.get_content(metadata_mode=MetadataMode.NONE) or "",
                "metadata": json.dumps(metadata),
            }
            for k, v in metadata.items():
                properties[PROPERTY_PREFIX + k] = v

            command = {
                "AddDescriptor": {
                    "class": self._descriptor_set,
                    "properties": properties,  ## Can add arbitrary key value pairs here.
                    "if_not_found": predicate(UNIQUEID_PROPERTY, "==", node.node_id),
                }
            }

            query = [command]
            blobs = [
                np.array(node.embedding, dtype=np.float32).tobytes()
            ]  ## And convert the already calculated embeddings into blobs here
            data.append((query, blobs))
            ids.append(node.node_id)

        loader = ParallelLoader(self._client)
        loader.ingest(data, batchsize=BATCHSIZE)
        if loader.error_counter:
            raise RuntimeError(
                f"Failed to ingest {loader.error_counter} descriptor batches"
            )

        return ids

    def delete_vector_store(self, descriptor_set_name: str) -> None:
        """
        Delete a descriptor set from ApertureDB.

        Args:
            descriptor_set_name: The name of the descriptor set to delete.

        """
        self._utils.remove_descriptorset(descriptor_set_name)

    def get_descriptor_set(self) -> List[str]:
        """
        Return a list of existing descriptor sets in ApertureDB.
        """
        return self._utils.get_descriptorset_list()

    def query(self, query: VectorStoreQuery) -> VectorStoreQueryResult:
        """
        Return nodes as response.
        """

        def convert_filters_to_constraints(
            query_filters: Optional[MetadataFilters],
        ) -> Optional[Constraints]:
            if query_filters is None:
                return None
            if query_filters.condition not in (
                None,
                FilterCondition.AND,
                FilterCondition.OR,
            ):
                raise ValueError(
                    f"Unsupported filter condition: {query_filters.condition}"
                )
            constraints = Constraints(
                Conjunction.OR
                if query_filters.condition == FilterCondition.OR
                else Conjunction.AND
            )
            for filter in query_filters.filters:
                key = f"{PROPERTY_PREFIX}{filter.key}"
                if filter.operator == FilterOperator.EQ:
                    constraints = constraints.equal(
                        key,
                        filter.value,
                    )
                elif filter.operator == FilterOperator.GT:
                    constraints = constraints.greater(
                        key,
                        filter.value,
                    )
                elif filter.operator == FilterOperator.LT:
                    constraints = constraints.less(
                        key,
                        filter.value,
                    )
                elif filter.operator == FilterOperator.GTE:
                    constraints = constraints.greaterequal(
                        key,
                        filter.value,
                    )
                elif filter.operator == FilterOperator.LTE:
                    constraints = constraints.lessequal(
                        key,
                        filter.value,
                    )
                else:
                    raise ValueError(f"Unsupported filter operator: {filter.operator}")
            return constraints

        constraints = convert_filters_to_constraints(query.filters)
        ## VectorStoreQuery has query_embedding, similarity_top_k and mode
        self._descriptors.find_similar(
            set=self._descriptor_set,
            constraints=constraints,
            vector=query.query_embedding,
            k_neighbors=query.similarity_top_k,
            distances=True,
            engine=self._engine,
            metric=self._metric,
        )

        nodes = []
        ids: List[str] = []
        similarities: List[float] = []

        for d in self._descriptors:
            metadata = json.loads(d["metadata"])
            node = metadata_dict_to_node(metadata)

            text = d.get("text")

            node.set_content(text)

            nodes.append(node)
            ids.append(d[UNIQUEID_PROPERTY])
            similarities.append(d["_distance"])

        return VectorStoreQueryResult(nodes=nodes, ids=ids, similarities=similarities)

    def delete(self, ref_doc_id: str, **delete_kwargs: Any) -> None:
        """
        Delete embeddings (if present) from the vectorstore confined the given ref_doc_id.
        They should additionally be confined to the descriptor set in use.

        Args:
            ref_doc_id: The document ID to delete.
            delete_kwargs: Additional arguments to pass to delete

        Returns:
            None

        """
        query = [
            {
                "DeleteDescriptor": {
                    "constraints": predicate(
                        PROPERTY_PREFIX + "ref_doc_id", "==", ref_doc_id
                    ),
                    "with_class": self._descriptor_set,
                }
            }
        ]

        result, _ = self._utils.execute(query)
        assert len(result) == 1, f"Failed to delete descriptor {result=}"
        assert result[0]["DeleteDescriptor"]["status"] == 0, (
            f"Failed to delete descriptor {result=}"
        )

    def clear(self) -> None:
        """
        Delete all descriptors in the specified descriptor set.
        """
        query = [
            {
                "DeleteDescriptor": {
                    "with_class": self._descriptor_set,
                }
            }
        ]

        result, _ = self._utils.execute(query)
        assert len(result) == 1, f"Failed to delete descriptor {result=}"
        assert result[0]["DeleteDescriptor"]["status"] == 0, (
            f"Failed to delete descriptor {result=}"
        )

    def delete_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
        **delete_kwargs: Any,
    ) -> List[str]:
        """Delete nodes from vector store."""
        return super().delete_nodes(node_ids, filters, **delete_kwargs)

    def get_nodes(
        self,
        node_ids: Optional[List[str]] = None,
        filters: Optional[MetadataFilters] = None,
    ) -> List[TextNode]:
        """Get nodes from vector store."""
        return super().get_nodes(node_ids, filters)
