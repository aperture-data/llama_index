import logging
from aperturedb.Constraints import predicate
from typing import Any, Dict, List, Optional
from llama_index.core.graph_stores.types import (
    LabelledNode,
    PropertyGraphStore,
    EntityNode,
    Relation,
    ChunkNode,
    Triplet,
)

query_executor = None
# Prefix for properties that are in the client metadata
PROPERTY_PREFIX = "lm_"

TEXT_PROPERTY = "text"  # Property name for the text
UNIQUEID_PROPERTY = "uniqueid"  # Property name for the unique id

BATCHSIZE = 1000

logger = logging.getLogger(__name__)


def get_entity(client, label, id):
    """Get entity by id."""
    query = [
        {
            "FindEntity": {
                "_ref": 1,
                "constraints": predicate(UNIQUEID_PROPERTY, "==", id),
                "results": {"all_properties": True},
            }
        }
    ]
    if label is not None:
        query[0]["FindEntity"]["with_class"] = label
    result, response, _ = query_executor(
        client,
        query,
    )
    assert result == 0, response
    if (
        "entities" in response[0]["FindEntity"]
        and len(response[0]["FindEntity"]["entities"]) > 0
    ):
        return response[0]["FindEntity"]["entities"][0]
    return None


def changed(entity, properties):
    """Check if properties have changed."""
    to_update = {}
    to_delete = []
    if entity is None:
        return properties, to_delete
    for k, v in properties.items():
        if k not in entity:
            to_update[k] = v
        elif entity[k] != v:
            to_update[k] = v
    for k, v in entity.items():
        if k not in properties and not k.startswith("_") and k not in ["id", "name"]:
            to_delete.append(k)
    return to_update, to_delete


def query_for_ids(command: str, ids: List[str]) -> List[dict]:
    """Create a query for a list of ids."""
    params = {"constraints": predicate(UNIQUEID_PROPERTY, "in", ids)}
    if command == "FindEntity":
        params["results"] = {"all_properties": True}
    return [{command: params}]


def query_for_properties(command: str, properties: dict) -> List[dict]:
    terms = [predicate(key, "==", value) for key, value in properties.items()]
    if command == "DeleteEntity":
        terms.extend(
            [
                ["not", ["$label", "is_missing"]],
                ["not", ["$" + UNIQUEID_PROPERTY, "is_missing"]],
            ]
        )
    params = {"constraints": ["all", *terms]}
    if command == "FindEntity":
        params["results"] = {"all_properties": True}
    return [{command: params}]


class ApertureDBGraphStore(PropertyGraphStore):
    """
    ApertureDB graph store.

    Args:
        config (dict): Configuration for the graph store.
        **kwargs: Additional keyword arguments.

    """

    flat_metadata: bool = True

    @property
    def client(self) -> Any:
        """Get client."""
        return self._client

    def __init__(self, *args, **kwargs) -> None:
        try:
            from aperturedb.CommonLibrary import create_connector, execute_query
        except ImportError:
            raise ImportError(
                "Install the ApertureDB SDK with pip install 'aperturedb>=1.0.3'."
            )

        self._client = create_connector()
        global query_executor
        query_executor = execute_query

    def get_rel_map(
        self,
        subjs: List[LabelledNode],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        frontier = list(subjs or [])
        visited = set()
        triplets = []
        for _ in range(depth):
            following = []
            for source in frontier:
                if source.id in visited or len(triplets) >= limit:
                    continue
                visited.add(source.id)
                query = [
                    {
                        "FindEntity": {
                            "with_class": source.label,
                            "_ref": 1,
                            "constraints": predicate(
                                UNIQUEID_PROPERTY, "==", source.id.capitalize()
                            ),
                        }
                    },
                    {"FindConnection": {"src": 1, "results": {"all_properties": True}}},
                ]
                result, response, _ = query_executor(self._client, query)
                assert result == 0, response
                edges = [
                    e
                    for e in response[1]["FindConnection"].get("connections", [])
                    if "dst_id" in e
                    and UNIQUEID_PROPERTY in e
                    and e[UNIQUEID_PROPERTY] not in (ignore_rels or [])
                ]
                if not edges:
                    continue
                targets = {n.id: n for n in self.get(ids=[e["dst_id"] for e in edges])}
                for edge in edges:
                    target = targets.get(edge["dst_id"])
                    if target is None:
                        continue
                    relation = Relation(
                        source_id=source.id,
                        target_id=target.id,
                        label=edge[UNIQUEID_PROPERTY],
                    )
                    triplets.append((source, relation, target))
                    following.append(target)
                    if len(triplets) >= limit:
                        return triplets
            frontier = following
            if not frontier:
                break
        return triplets

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete nodes."""
        if ids and len(ids) > 0:
            query = query_for_ids("DeleteEntity", [id.capitalize() for id in ids])
            result, response, _ = query_executor(
                self._client,
                query,
            )
            assert result == 0, response
        if properties and len(properties) > 0:
            query = query_for_properties("DeleteEntity", properties)
            result, response, _ = query_executor(
                self._client,
                query,
            )
            assert result == 0, response
        if entity_names and len(entity_names) > 0:
            for name in entity_names:
                query = [
                    {
                        "DeleteEntity": {
                            "with_class": name,
                            "constraints": predicate("label", "==", name),
                        }
                    }
                ]
                result, response, _ = query_executor(
                    self._client,
                    query,
                )
                assert result == 0, response
        if relation_names and len(relation_names) > 0:
            for relation_name in set(relation_names):
                query = [
                    {
                        "DeleteConnection": {
                            "with_class": relation_name,
                            "constraints": predicate(
                                UNIQUEID_PROPERTY, "==", relation_name
                            ),
                        }
                    }
                ]
                result, response, _ = query_executor(
                    self._client,
                    query,
                )
                assert result == 0, response

    def get(
        self, properties: Optional[dict] = None, ids: Optional[List[str]] = None
    ) -> List[LabelledNode]:
        if ids:
            query = query_for_ids("FindEntity", [id.capitalize() for id in ids])
        elif properties:
            query = query_for_properties("FindEntity", properties)
        else:
            query = [
                {
                    "FindEntity": {
                        "constraints": [
                            "all",
                            ["not", ["$label", "is_missing"]],
                            ["not", ["$" + UNIQUEID_PROPERTY, "is_missing"]],
                        ],
                        "limit": BATCHSIZE,
                        "results": {"all_properties": True},
                    }
                }
            ]
        result, response, _ = query_executor(self._client, query)
        assert result == 0, response
        nodes = []
        for entity in response[0]["FindEntity"].get("entities", []):
            if "label" not in entity or UNIQUEID_PROPERTY not in entity:
                continue
            if entity["label"] == "text_chunk":
                nodes.append(
                    ChunkNode(
                        properties={
                            "_" + k: entity[k]
                            for k in ("node_content", "node_type")
                            if k in entity
                        },
                        text=entity["text"],
                        id_=entity[UNIQUEID_PROPERTY],
                    )
                )
            else:
                nodes.append(
                    EntityNode(
                        label=entity["label"],
                        properties=entity,
                        name=entity[UNIQUEID_PROPERTY],
                    )
                )
        return nodes

    def get_triplets(
        self, entity_names=None, relation_names=None, properties=None, ids=None
    ):
        raise NotImplementedError("get_triplets is not implemented")

    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        query = [{query: param_map}]
        blobs = []
        result, response, _ = query_executor(self._client, query, blobs)
        assert result == 0, response
        return response

    def upsert_nodes(self, nodes: List[EntityNode]) -> List[str]:
        ids = []

        for node in nodes:
            properties = node.properties
            id = node.id.capitalize()
            if isinstance(node, ChunkNode):
                sane_props = {
                    "text": node.text,
                }
                for k, v in node.properties.items():
                    if k.startswith("_"):
                        sane_props[k[1:]] = v
                properties = sane_props

            entity = get_entity(self._client, node.label, id)
            combined_properties = properties | {
                UNIQUEID_PROPERTY: id,
                "label": node.label,
            }

            command = None
            if entity is None:
                command = {
                    "AddEntity": {
                        "class": node.label,
                        "if_not_found": predicate(UNIQUEID_PROPERTY, "==", id),
                        "properties": combined_properties,
                    }
                }
            else:
                to_update, to_delete = changed(entity, combined_properties)
                if len(to_update) > 0 or len(to_delete) > 0:
                    command = {
                        "UpdateEntity": {
                            "with_class": node.label,
                            "constraints": predicate(UNIQUEID_PROPERTY, "==", id),
                            "properties": to_update,
                            "remove_props": to_delete,
                        }
                    }

            if command is not None:
                query = [command]
                blobs = []
                result, response, _ = query_executor(self._client, query, blobs)
                assert result == 0, response
                ids.append(id)

        return ids

    def upsert_relations(self, relations: List[Relation]) -> None:
        """Upsert relations."""
        ids = []
        for r in relations:
            query = [
                {
                    "FindEntity": {
                        "constraints": predicate(
                            UNIQUEID_PROPERTY, "==", r.source_id.capitalize()
                        ),
                        "_ref": 1,
                    }
                },
                {
                    "FindEntity": {
                        "constraints": predicate(
                            UNIQUEID_PROPERTY, "==", r.target_id.capitalize()
                        ),
                        "_ref": 2,
                    }
                },
                {
                    "AddConnection": {
                        "class": r.label,
                        "src": 1,
                        "dst": 2,
                        "properties": r.properties
                        | {
                            UNIQUEID_PROPERTY: f"{r.id}",
                            "src_id": r.source_id.capitalize(),
                            "dst_id": r.target_id.capitalize(),
                        },
                        "if_not_found": [
                            "all",
                            predicate(UNIQUEID_PROPERTY, "==", r.id),
                            predicate("src_id", "==", r.source_id.capitalize()),
                            predicate("dst_id", "==", r.target_id.capitalize()),
                        ],
                    }
                },
            ]
            result, response, _ = query_executor(
                self._client, query, success_statuses=[0, 2]
            )
            assert result == 0, response
            ids.append(r.id)
        return ids

    def vector_query(self, query, **kwargs):
        raise NotImplementedError("vector_query is not implemented")
