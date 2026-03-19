"""
Integration tests for the full project workflow through QdrantMCPServer MCP tools.

Tests use in-memory Qdrant + FastEmbed + temp directories for project config.
Tools are invoked via fastmcp.Client to test the complete end-to-end flow.
"""
import json
import uuid

import pytest
from fastmcp import Client

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.qdrant import QdrantConnector
from mcp_server_qdrant.settings import (
    ProjectSettings,
    QdrantSettings,
    ToolSettings,
    load_project_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def embedding_provider():
    return FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
async def tmp_project_dir(tmp_path):
    """Temp directory to serve as project dir (holds .qdrant-project.json)."""
    return tmp_path


@pytest.fixture
async def qdrant_connector(embedding_provider):
    """In-memory QdrantConnector with no default collection."""
    return QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=None,
        embedding_provider=embedding_provider,
    )


@pytest.fixture
async def mcp_server(tmp_project_dir, embedding_provider):
    """
    QdrantMCPServer with in-memory Qdrant, FastEmbed, and project dir set to tmp_project_dir.
    """
    qdrant_settings = QdrantSettings.model_validate({"QDRANT_URL": ":memory:"})
    project_settings = ProjectSettings.model_validate(
        {"QDRANT_PROJECT_DIR": str(tmp_project_dir)}
    )
    return QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=qdrant_settings,
        embedding_provider=embedding_provider,
        project_settings=project_settings,
    )


# ---------------------------------------------------------------------------
# 1. Init project creates config and collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_project_creates_config_and_collection(mcp_server, tmp_project_dir):
    """Init project: .qdrant-project.json created and collection exists in Qdrant."""
    project_name = f"myproject_{uuid.uuid4().hex[:6]}"

    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "qdrant-init-project", {"project_name": project_name}
        )

    text = result[0].text
    assert project_name in text
    assert "initialized" in text.lower()

    # Response contains config JSON for client to persist
    assert ".qdrant-project.json" in text
    assert f"proj_{project_name}" in text

    # Parse the JSON block from the response
    json_start = text.index("```json\n") + len("```json\n")
    json_end = text.index("\n```", json_start)
    config = json.loads(text[json_start:json_end])
    assert config["project_name"] == project_name
    assert config["collection"] == f"proj_{project_name}"
    assert config["linked_collections"] == []

    # Collection exists in Qdrant
    assert await mcp_server.qdrant_connector._client.collection_exists(
        f"proj_{project_name}"
    )


# ---------------------------------------------------------------------------
# 2. Init project returns existing collections for linking
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_project_returns_existing_collections(mcp_server):
    """Init project when other collections already exist: response lists them."""
    extra1 = f"extra1_{uuid.uuid4().hex[:6]}"
    extra2 = f"extra2_{uuid.uuid4().hex[:6]}"
    await mcp_server.qdrant_connector.ensure_collection_exists(extra1)
    await mcp_server.qdrant_connector.ensure_collection_exists(extra2)

    project_name = f"proj_{uuid.uuid4().hex[:6]}"

    async with Client(mcp_server) as client:
        result = await client.call_tool(
            "qdrant-init-project", {"project_name": project_name}
        )

    text = result[0].text
    assert extra1 in text
    assert extra2 in text


# ---------------------------------------------------------------------------
# 3. Init project default collection name
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_project_default_collection_name(mcp_server, tmp_project_dir):
    """Init with only project_name: collection is proj_{name}."""
    project_name = f"myapp_{uuid.uuid4().hex[:6]}"

    async with Client(mcp_server) as client:
        result = await client.call_tool("qdrant-init-project", {"project_name": project_name})

    text = result[0].text
    json_start = text.index("```json\n") + len("```json\n")
    json_end = text.index("\n```", json_start)
    config = json.loads(text[json_start:json_end])
    assert config["collection"] == f"proj_{project_name}"


# ---------------------------------------------------------------------------
# 4. Init project already exists returns error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_init_project_already_exists_error(mcp_server):
    """Init project twice: second call returns an already-initialized message.

    Note: since the server no longer writes to disk, the "already initialized"
    check is based on in-memory state (self.project_config).
    """
    project_name = f"dup_{uuid.uuid4().hex[:6]}"

    async with Client(mcp_server) as client:
        await client.call_tool("qdrant-init-project", {"project_name": project_name})
        result2 = await client.call_tool(
            "qdrant-init-project", {"project_name": project_name}
        )

    assert "already initialized" in result2[0].text.lower()


# ---------------------------------------------------------------------------
# 5. Link collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_link_collection(mcp_server, tmp_project_dir):
    """Init project, create knowledge collection, link it — config updated."""
    project_name = f"proj_{uuid.uuid4().hex[:6]}"
    knowledge_col = f"knowledge_{uuid.uuid4().hex[:6]}"

    await mcp_server.qdrant_connector.ensure_collection_exists(knowledge_col)

    async with Client(mcp_server) as client:
        await client.call_tool("qdrant-init-project", {"project_name": project_name})
        result = await client.call_tool(
            "qdrant-link", {"collection_name": knowledge_col}
        )

    text = result[0].text
    assert knowledge_col in text
    assert "linked" in text.lower()

    # Response contains updated config JSON for client to persist
    assert ".qdrant-project.json" in text
    json_start = text.index("```json\n") + len("```json\n")
    json_end = text.index("\n```", json_start)
    config = json.loads(text[json_start:json_end])
    assert knowledge_col in config["linked_collections"]


# ---------------------------------------------------------------------------
# 6. Link nonexistent collection returns error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_link_nonexistent_collection_error(mcp_server):
    """Try to link a collection that doesn't exist — returns error message."""
    project_name = f"proj_{uuid.uuid4().hex[:6]}"
    nonexistent = f"nonexistent_{uuid.uuid4().hex[:6]}"

    async with Client(mcp_server) as client:
        await client.call_tool("qdrant-init-project", {"project_name": project_name})
        result = await client.call_tool(
            "qdrant-link", {"collection_name": nonexistent}
        )

    assert "does not exist" in result[0].text.lower()


# ---------------------------------------------------------------------------
# 7. Link duplicate is idempotent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_link_duplicate_is_idempotent(mcp_server, tmp_project_dir):
    """Link the same collection twice — no duplicate in config."""
    project_name = f"proj_{uuid.uuid4().hex[:6]}"
    col = f"col_{uuid.uuid4().hex[:6]}"

    await mcp_server.qdrant_connector.ensure_collection_exists(col)

    async with Client(mcp_server) as client:
        await client.call_tool("qdrant-init-project", {"project_name": project_name})
        await client.call_tool("qdrant-link", {"collection_name": col})
        await client.call_tool("qdrant-link", {"collection_name": col})

    # In-memory config should have no duplicate
    assert mcp_server.project_config is not None
    assert mcp_server.project_config.linked_collections.count(col) == 1


# ---------------------------------------------------------------------------
# 8. Unlink collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unlink_collection(mcp_server, tmp_project_dir):
    """Link then unlink a collection — config updated to remove it."""
    project_name = f"proj_{uuid.uuid4().hex[:6]}"
    col = f"col_{uuid.uuid4().hex[:6]}"

    await mcp_server.qdrant_connector.ensure_collection_exists(col)

    async with Client(mcp_server) as client:
        await client.call_tool("qdrant-init-project", {"project_name": project_name})
        await client.call_tool("qdrant-link", {"collection_name": col})
        result = await client.call_tool("qdrant-unlink", {"collection_name": col})

    text = result[0].text
    assert "unlinked" in text.lower()

    # Response contains updated config JSON for client to persist
    assert ".qdrant-project.json" in text
    json_start = text.index("```json\n") + len("```json\n")
    json_end = text.index("\n```", json_start)
    config = json.loads(text[json_start:json_end])
    assert col not in config["linked_collections"]


# ---------------------------------------------------------------------------
# 9. Unlink not-linked collection is handled gracefully
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unlink_not_linked_collection(mcp_server):
    """Unlink a collection that was never linked — graceful error message."""
    project_name = f"proj_{uuid.uuid4().hex[:6]}"
    col = f"col_{uuid.uuid4().hex[:6]}"

    async with Client(mcp_server) as client:
        await client.call_tool("qdrant-init-project", {"project_name": project_name})
        result = await client.call_tool("qdrant-unlink", {"collection_name": col})

    assert "not linked" in result[0].text.lower()


# ---------------------------------------------------------------------------
# 10. Store defaults to project collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_defaults_to_project_collection(mcp_server):
    """Init project, store without explicit collection_name — stored in project collection."""
    project_name = f"proj_{uuid.uuid4().hex[:6]}"
    content = "The quick brown fox jumps over the lazy dog"
    project_col = f"proj_{project_name}"

    async with Client(mcp_server) as client:
        await client.call_tool("qdrant-init-project", {"project_name": project_name})
        await client.call_tool(
            "qdrant-store", {"information": content, "collection_name": ""}
        )

    results = await mcp_server.qdrant_connector.search(
        "fox jumps", collection_name=project_col
    )
    assert len(results) == 1
    assert results[0].content == content


# ---------------------------------------------------------------------------
# 11. Store with explicit collection overrides project collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_store_explicit_collection_override(mcp_server):
    """Init project, store with explicit collection_name — stored in that collection."""
    project_name = f"proj_{uuid.uuid4().hex[:6]}"
    explicit_col = f"explicit_{uuid.uuid4().hex[:6]}"
    content = "Explicit collection content here"
    project_col = f"proj_{project_name}"

    async with Client(mcp_server) as client:
        await client.call_tool("qdrant-init-project", {"project_name": project_name})
        await client.call_tool(
            "qdrant-store",
            {"information": content, "collection_name": explicit_col},
        )

    # In explicit collection
    results = await mcp_server.qdrant_connector.search(
        "explicit", collection_name=explicit_col
    )
    assert len(results) == 1

    # Not in project collection
    results_proj = await mcp_server.qdrant_connector.search(
        "explicit", collection_name=project_col
    )
    assert len(results_proj) == 0


# ---------------------------------------------------------------------------
# 12. Find searches project and linked collections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_searches_project_and_linked(mcp_server):
    """Init project, link collection, store in both, find without collection_name returns both."""
    project_name = f"proj_{uuid.uuid4().hex[:6]}"
    linked_col = f"knowledge_{uuid.uuid4().hex[:6]}"
    project_col = f"proj_{project_name}"

    await mcp_server.qdrant_connector.ensure_collection_exists(linked_col)

    async with Client(mcp_server) as client:
        await client.call_tool("qdrant-init-project", {"project_name": project_name})
        await client.call_tool("qdrant-link", {"collection_name": linked_col})

        await client.call_tool(
            "qdrant-store",
            {
                "information": "Python is a great programming language",
                "collection_name": project_col,
            },
        )
        await client.call_tool(
            "qdrant-store",
            {
                "information": "Python web frameworks include Django and Flask",
                "collection_name": linked_col,
            },
        )

        # Search without collection_name — fans out to both
        find_result = await client.call_tool(
            "qdrant-find", {"query": "Python", "collection_name": ""}
        )

    text = " ".join(r.text for r in find_result)
    assert project_col in text
    assert linked_col in text


# ---------------------------------------------------------------------------
# 13. Find with explicit collection searches only that collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_explicit_collection_searches_single(mcp_server):
    """Search with explicit collection_name — only that collection searched."""
    project_name = f"proj_{uuid.uuid4().hex[:6]}"
    linked_col = f"linked_{uuid.uuid4().hex[:6]}"
    project_col = f"proj_{project_name}"

    await mcp_server.qdrant_connector.ensure_collection_exists(linked_col)

    async with Client(mcp_server) as client:
        await client.call_tool("qdrant-init-project", {"project_name": project_name})
        await client.call_tool("qdrant-link", {"collection_name": linked_col})

        await client.call_tool(
            "qdrant-store",
            {"information": "Project data about Python", "collection_name": project_col},
        )
        await client.call_tool(
            "qdrant-store",
            {"information": "Linked data about Python", "collection_name": linked_col},
        )

        # Search with explicit project_col only
        find_result = await client.call_tool(
            "qdrant-find", {"query": "Python", "collection_name": project_col}
        )

    text = " ".join(r.text for r in find_result)
    assert "Project data" in text
    # linked_col name should not appear (no fan-out)
    assert linked_col not in text


# ---------------------------------------------------------------------------
# 14. Project info shows stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_project_info_shows_stats(mcp_server):
    """Init project, store entries, qdrant-project-info shows stats."""
    project_name = f"proj_{uuid.uuid4().hex[:6]}"
    project_col = f"proj_{project_name}"

    async with Client(mcp_server) as client:
        await client.call_tool("qdrant-init-project", {"project_name": project_name})

        for i in range(3):
            await client.call_tool(
                "qdrant-store",
                {
                    "information": f"Document number {i} about science",
                    "collection_name": project_col,
                },
            )

        result = await client.call_tool("qdrant-project-info", {})

    text = result[0].text
    assert project_name in text
    assert project_col in text
    assert "3" in text


# ---------------------------------------------------------------------------
# 15. Backward compat: no config, env-var-driven collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_backward_compat_no_config(embedding_provider, tmp_project_dir):
    """Without .qdrant-project.json, store/find work as before with fixed collection."""
    collection_name = f"compat_{uuid.uuid4().hex[:6]}"

    # Server with a fixed collection_name and no project_settings
    server = QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings.model_validate(
            {"QDRANT_URL": ":memory:", "COLLECTION_NAME": collection_name}
        ),
        embedding_provider=embedding_provider,
        project_settings=None,
    )

    content = "Standard backward-compatible entry"

    async with Client(server) as client:
        # When collection_name is set in QdrantSettings, it's baked in via make_partial_function
        # so we do not pass collection_name as an argument here
        await client.call_tool("qdrant-store", {"information": content})
        find_result = await client.call_tool(
            "qdrant-find", {"query": "backward compatible"}
        )

    text = " ".join(r.text for r in find_result)
    assert "Standard" in text or "backward-compatible" in text

    # No config file was created
    assert load_project_config(tmp_project_dir) is None
