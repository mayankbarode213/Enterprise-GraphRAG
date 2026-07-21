"""
pytest configuration — enables asyncio mode for all tests.
"""
import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "integration: marks integration tests (require Neo4j/FAISS)")


@pytest.fixture(scope="function", autouse=True)
async def cleanup_neo4j_each_test():
    yield
    import app.graph.client as client_module
    if client_module._client is not None:
        try:
            await client_module._client.close()
        except Exception:
            pass
        client_module._client = None
