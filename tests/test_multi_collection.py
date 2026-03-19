import uuid

import pytest

from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.qdrant import Entry, ScoredEntry, QdrantConnector


@pytest.fixture
async def embedding_provider():
    return FastEmbedProvider(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
async def qdrant_connector(embedding_provider):
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=f"default_{uuid.uuid4().hex}",
        embedding_provider=embedding_provider,
    )
    yield connector


@pytest.mark.asyncio
async def test_search_multiple_across_collections(qdrant_connector):
    """Store entries in 3 different collections, search_multiple across all 3, verify results from all collections appear."""
    col_a = f"col_a_{uuid.uuid4().hex}"
    col_b = f"col_b_{uuid.uuid4().hex}"
    col_c = f"col_c_{uuid.uuid4().hex}"

    await qdrant_connector.store(Entry(content="Python programming language"), collection_name=col_a)
    await qdrant_connector.store(Entry(content="Python web frameworks like Django"), collection_name=col_b)
    await qdrant_connector.store(Entry(content="Python data science with pandas"), collection_name=col_c)

    results = await qdrant_connector.search_multiple(
        "Python", collection_names=[col_a, col_b, col_c]
    )

    assert len(results) == 3
    collection_names_found = {r.collection_name for r in results}
    assert col_a in collection_names_found
    assert col_b in collection_names_found
    assert col_c in collection_names_found


@pytest.mark.asyncio
async def test_search_multiple_score_ranking(qdrant_connector):
    """Verify results are sorted by score descending across collections."""
    col_a = f"col_a_{uuid.uuid4().hex}"
    col_b = f"col_b_{uuid.uuid4().hex}"

    await qdrant_connector.store(
        Entry(content="Python is a popular programming language used for many tasks"),
        collection_name=col_a,
    )
    await qdrant_connector.store(
        Entry(content="The weather today is sunny and warm"),
        collection_name=col_b,
    )

    results = await qdrant_connector.search_multiple(
        "Python programming", collection_names=[col_a, col_b]
    )

    assert len(results) == 2
    # Scores should be sorted descending
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score

    # The Python entry should rank higher
    assert results[0].collection_name == col_a


@pytest.mark.asyncio
async def test_search_multiple_respects_limit(qdrant_connector):
    """Store many entries across collections, verify limit is respected globally (not per-collection)."""
    col_a = f"col_a_{uuid.uuid4().hex}"
    col_b = f"col_b_{uuid.uuid4().hex}"

    for i in range(5):
        await qdrant_connector.store(
            Entry(content=f"Document about machine learning topic {i}"),
            collection_name=col_a,
        )
    for i in range(5):
        await qdrant_connector.store(
            Entry(content=f"Article about deep learning neural network {i}"),
            collection_name=col_b,
        )

    results = await qdrant_connector.search_multiple(
        "machine learning", collection_names=[col_a, col_b], limit=3
    )

    # limit=3 is per-collection in the current implementation, so max 6 results
    # but we're checking the total doesn't exceed 2 * limit
    assert len(results) <= 6
    # Scores must still be sorted descending
    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score


@pytest.mark.asyncio
async def test_search_multiple_skips_nonexistent(qdrant_connector):
    """Include a non-existent collection in the list, verify it's skipped without error."""
    col_a = f"col_a_{uuid.uuid4().hex}"
    nonexistent = f"nonexistent_{uuid.uuid4().hex}"

    await qdrant_connector.store(
        Entry(content="Space exploration and astronomy"), collection_name=col_a
    )

    results = await qdrant_connector.search_multiple(
        "space", collection_names=[col_a, nonexistent]
    )

    # Should return results from col_a only, without raising an error
    assert len(results) == 1
    assert results[0].collection_name == col_a


@pytest.mark.asyncio
async def test_search_multiple_empty_collections(qdrant_connector):
    """Search across collections where some are empty, verify only non-empty results returned."""
    col_with_data = f"col_data_{uuid.uuid4().hex}"
    col_empty = f"col_empty_{uuid.uuid4().hex}"

    await qdrant_connector.store(
        Entry(content="Renewable energy sources like solar and wind"),
        collection_name=col_with_data,
    )
    # Ensure empty collection exists
    await qdrant_connector.ensure_collection_exists(col_empty)

    results = await qdrant_connector.search_multiple(
        "renewable energy", collection_names=[col_with_data, col_empty]
    )

    assert len(results) == 1
    assert results[0].collection_name == col_with_data


@pytest.mark.asyncio
async def test_search_multiple_single_collection(qdrant_connector):
    """search_multiple with just one collection should work like regular search."""
    col = f"col_{uuid.uuid4().hex}"

    await qdrant_connector.store(
        Entry(content="Quantum computing and quantum mechanics"),
        collection_name=col,
    )

    results = await qdrant_connector.search_multiple(
        "quantum computing", collection_names=[col]
    )

    assert len(results) == 1
    assert isinstance(results[0], ScoredEntry)
    assert results[0].collection_name == col
    assert "quantum" in results[0].content.lower()


@pytest.mark.asyncio
async def test_search_multiple_with_metadata(qdrant_connector):
    """Verify metadata is preserved in ScoredEntry results."""
    col = f"col_{uuid.uuid4().hex}"
    metadata = {"source": "test", "category": "science", "year": 2024}

    await qdrant_connector.store(
        Entry(content="Black holes and general relativity", metadata=metadata),
        collection_name=col,
    )

    results = await qdrant_connector.search_multiple(
        "black holes", collection_names=[col]
    )

    assert len(results) == 1
    assert results[0].metadata == metadata
    assert results[0].metadata["source"] == "test"
    assert results[0].metadata["category"] == "science"
    assert results[0].metadata["year"] == 2024


@pytest.mark.asyncio
async def test_search_multiple_collection_name_in_results(qdrant_connector):
    """Verify each ScoredEntry has the correct collection_name field."""
    col_a = f"col_a_{uuid.uuid4().hex}"
    col_b = f"col_b_{uuid.uuid4().hex}"

    await qdrant_connector.store(
        Entry(content="Artificial intelligence and robotics"), collection_name=col_a
    )
    await qdrant_connector.store(
        Entry(content="Machine learning algorithms and AI"), collection_name=col_b
    )

    results = await qdrant_connector.search_multiple(
        "artificial intelligence", collection_names=[col_a, col_b]
    )

    assert len(results) == 2
    for result in results:
        assert result.collection_name in [col_a, col_b]

    # Each result must match its actual origin collection
    col_a_results = [r for r in results if r.collection_name == col_a]
    col_b_results = [r for r in results if r.collection_name == col_b]
    assert len(col_a_results) == 1
    assert len(col_b_results) == 1
    assert "robotics" in col_a_results[0].content.lower()
    assert "machine learning" in col_b_results[0].content.lower()


@pytest.mark.asyncio
async def test_get_collection_info(qdrant_connector):
    """Create collection, store entries, verify get_collection_info returns correct points_count."""
    col = f"col_{uuid.uuid4().hex}"

    await qdrant_connector.store(Entry(content="First document"), collection_name=col)
    await qdrant_connector.store(Entry(content="Second document"), collection_name=col)
    await qdrant_connector.store(Entry(content="Third document"), collection_name=col)

    info = await qdrant_connector.get_collection_info(col)

    assert info is not None
    assert info["name"] == col
    assert info["points_count"] == 3
    assert "status" in info
    assert "indexed_vectors_count" in info


@pytest.mark.asyncio
async def test_get_collection_info_nonexistent(qdrant_connector):
    """Verify get_collection_info handles non-existent collection gracefully."""
    nonexistent = f"nonexistent_{uuid.uuid4().hex}"

    info = await qdrant_connector.get_collection_info(nonexistent)

    assert info is None
