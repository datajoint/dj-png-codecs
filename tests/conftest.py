import logging
import subprocess
from collections.abc import Generator
from typing import TypedDict

import datajoint as dj
import pytest
from testcontainers.community.mysql import MySqlContainer

logger = logging.getLogger(__name__)


def _is_docker_available() -> bool:
    """Check if Docker daemon is running and accessible."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    else:
        return result.returncode == 0


# Check if Docker is available
DOCKER_AVAILABLE = _is_docker_available()


class DBCreds(TypedDict):
    host: str
    user: str
    password: str


@pytest.fixture(scope="session")
def mysql_container() -> Generator[MySqlContainer, None, None]:
    """Start MySQL container for the test session."""
    if not DOCKER_AVAILABLE:
        pytest.skip("Docker is not available")
    container = MySqlContainer(
        image="mysql:8.0",
        username="root",
        password="password",
        dbname="test",
    )
    container.start()

    host = container.get_container_host_ip()
    port = container.get_exposed_port(3306)
    logger.info("MySQL container started at %s:%s", host, port)

    yield container

    container.stop()
    logger.info("MySQL container stopped")


@pytest.fixture(scope="session")
def db_creds(mysql_container: MySqlContainer) -> DBCreds:
    host = mysql_container.get_container_host_ip()
    port = mysql_container.get_exposed_port(3306)

    return {
        "host": f"{host}:{port}",
        "user": "root",
        "password": "password",
    }


@pytest.fixture
def dj_connection(db_creds: DBCreds, tmp_path) -> Generator[dj.Connection, None, None]:
    """Create connection for the specified backend.

    This fixture is function-scoped to ensure database config
    is restored after each test, preventing config pollution between tests.
    """
    # Save original config values
    old_config = {
        "database.host": dj.config.database.host,
        "database.port": dj.config.database.port,
        "database.user": dj.config.database.user,
        "database.password": dj.config.database.password,
        "database.use_tls": dj.config.database.use_tls,
        "safemode": dj.config.safemode,
        "loglevel": dj.config.loglevel,
    }
    old_stores = dict(dj.config.stores)

    # Parse host:port
    host_port = db_creds["host"]
    host, port = host_port.rsplit(":", 1)

    dj.config["database.host"] = host
    dj.config["database.port"] = int(port)
    dj.config["database.user"] = db_creds["user"]
    dj.config["database.password"] = db_creds["password"]
    dj.config["database.use_tls"] = False
    dj.config["safemode"] = False
    dj.config["loglevel"] = "INFO"

    # Configure a local file store for testing
    dj.config.stores["store"] = {
        "protocol": "file",
        "location": str(tmp_path),
    }

    logger.info("Connecting to MySQL at %s", host_port)

    connection = dj.Connection(
        host=host_port,
        user=db_creds["user"],
        password=db_creds["password"],
    )

    yield connection

    # Restore original config
    connection.close()
    for key, value in old_config.items():
        dj.config[key] = value
    dj.config.stores.clear()
    dj.config.stores.update(old_stores)


@pytest.fixture
def schema(dj_connection: dj.Connection) -> dj.Schema:
    """
    Fixture to create and drop a test schema for each test function.
    """
    test_schema = dj.Schema("test_schema", connection=dj_connection)
    yield test_schema
    test_schema.drop(prompt=False)
