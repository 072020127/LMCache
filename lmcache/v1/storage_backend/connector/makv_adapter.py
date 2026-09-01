# SPDX-License-Identifier: Apache-2.0

# First Party
from lmcache.logging import init_logger
from lmcache.v1.storage_backend.connector import ConnectorAdapter, ConnectorContext
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector

logger = init_logger(__name__)


class MaKVConnectorAdapter(ConnectorAdapter):
    """Adapter for MaKV remote manager connector."""

    def __init__(self) -> None:
        super().__init__("makv://")

    def create_connector(self, context: ConnectorContext) -> RemoteConnector:
        # Local
        from .makv_network_connector import MaKVNetworkConnector

        logger.info("Creating MaKV connector for URL: %s", context.url)
        return MaKVNetworkConnector(
            url=context.url,
            loop=context.loop,
            local_cpu_backend=context.local_cpu_backend,
            config=context.config,
            metadata=context.metadata,
        )
