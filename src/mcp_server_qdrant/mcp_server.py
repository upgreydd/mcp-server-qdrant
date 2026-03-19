import json
import logging
import re
from pathlib import Path
from typing import Annotated, Any, Optional

from fastmcp import Context, FastMCP
from pydantic import Field
from qdrant_client import models

from mcp_server_qdrant.common.filters import make_indexes
from mcp_server_qdrant.common.func_tools import make_partial_function
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector, ScoredEntry
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    ProjectConfig,
    ProjectSettings,
    QdrantSettings,
    ToolSettings,
    load_project_config,
)

logger = logging.getLogger(__name__)

_VALID_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")


def _validate_name(name: str, label: str = "name") -> str | None:
    """Validate a collection or project name. Returns an error message or None if valid."""
    if not name or not _VALID_NAME_PATTERN.match(name):
        return (
            f"Invalid {label}: '{name}'. "
            "Must be 1-128 characters, alphanumeric, hyphens, and underscores only, "
            "starting with an alphanumeric character."
        )
    return None


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        project_settings: Optional[ProjectSettings] = None,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings
        self.project_settings = project_settings

        if embedding_provider_settings and embedding_provider:
            raise ValueError(
                "Cannot provide both embedding_provider_settings and embedding_provider"
            )

        if not embedding_provider_settings and not embedding_provider:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )

        self.embedding_provider_settings: Optional[EmbeddingProviderSettings] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        else:
            self.embedding_provider_settings = None
            self.embedding_provider = embedding_provider

        assert self.embedding_provider is not None, "Embedding provider is required"

        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.local_path,
            make_indexes(qdrant_settings.filterable_fields_dict()),
        )

        # Load project config if project_settings is provided
        if project_settings is not None:
            self.project_config: Optional[ProjectConfig] = load_project_config(
                project_settings.resolved_project_dir
            )
        else:
            self.project_config = None

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    def _get_project_dir(self) -> Path | None:
        if self.project_settings is not None:
            return self.project_settings.resolved_project_dir
        return None

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def format_scored_entry(self, entry: ScoredEntry) -> str:
        """
        Format a ScoredEntry showing which collection it came from.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return (
            f"<entry><collection>{entry.collection_name}</collection>"
            f"<content>{entry.content}</content>"
            f"<metadata>{entry_metadata}</metadata>"
            f"<score>{entry.score:.4f}</score></entry>"
        )

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[
                str, Field(description="The collection to store the information in")
            ],
            # The `metadata` parameter is defined as non-optional, but it can be None.
            # If we set it to be optional, some of the MCP clients, like Cursor, cannot
            # handle the optional parameter correctly.
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="Extra metadata stored along with memorised information. Any json is accepted."
                ),
            ] = None,
        ) -> str:
            """
            Store some information in Qdrant.
            :param ctx: The context for the request.
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored.
            """
            await ctx.debug(f"Storing information {information} in Qdrant")

            # Resolve collection_name: explicit param > project config > env var
            resolved_collection = collection_name
            if not resolved_collection and self.project_config is not None:
                resolved_collection = self.project_config.collection

            entry = Entry(content=information, metadata=metadata)

            await self.qdrant_connector.store(entry, collection_name=resolved_collection)
            if resolved_collection:
                return f"Remembered: {information} in collection {resolved_collection}"
            return f"Remembered: {information}"

        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str, Field(description="The collection to search in")
            ],
            query_filter: ArbitraryFilter | None = None,
        ) -> list[str] | None:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :param query_filter: The filter to apply to the query.
            :return: A list of entries found or None.
            """

            # Log query_filter
            await ctx.debug(f"Query filter: {query_filter}")

            query_filter = models.Filter(**query_filter) if query_filter else None

            await ctx.debug(f"Finding results for query {query}")

            # When no explicit collection_name and project config exists, fan out to all project collections
            if not collection_name and self.project_config is not None:
                collections_to_search = [
                    self.project_config.collection
                ] + self.project_config.linked_collections

                scored_entries = await self.qdrant_connector.search_multiple(
                    query,
                    collection_names=collections_to_search,
                    limit=self.qdrant_settings.search_limit,
                    query_filter=query_filter,
                )
                if not scored_entries:
                    return None
                content = [f"Results for the query '{query}'"]
                for entry in scored_entries:
                    content.append(self.format_scored_entry(entry))
                return content

            # Fall back to single-collection search
            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
                query_filter=query_filter,
            )
            if not entries:
                return None
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        find_foo = find
        store_foo = store

        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})

        if self.qdrant_settings.collection_name:
            find_foo = make_partial_function(
                find_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            store_foo = make_partial_function(
                store_foo, {"collection_name": self.qdrant_settings.collection_name}
            )

        self.tool(
            find_foo,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database
            self.tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )

        # --- Project-aware tools ---

        async def qdrant_init_project(
            ctx: Context,
            project_name: Annotated[str, Field(description="The name of the project")],
            collection_name: Annotated[
                str | None,
                Field(
                    description="The Qdrant collection to use for this project. Defaults to 'proj_<project_name>'."
                ),
            ] = None,
        ) -> str:
            """
            Initialize a Qdrant project by creating a .qdrant-project.json config file.
            Creates the associated Qdrant collection if it does not already exist.
            """
            # Validate inputs
            err = _validate_name(project_name, "project name")
            if err:
                return err
            if collection_name is not None:
                err = _validate_name(collection_name, "collection name")
                if err:
                    return err

            project_dir = self._get_project_dir()

            # Check if already initialized (in-memory or on disk)
            existing = self.project_config or load_project_config(project_dir)
            if existing is not None:
                return (
                    f"Project already initialized: '{existing.project_name}' "
                    f"(collection: {existing.collection}). "
                    "Use qdrant-project-info to see current state."
                )

            resolved_collection = collection_name or f"proj_{project_name}"

            # Create the collection in Qdrant
            await self.qdrant_connector.ensure_collection_exists(resolved_collection)

            # Update in-memory config
            config = ProjectConfig(
                project_name=project_name,
                collection=resolved_collection,
                linked_collections=[],
            )
            self.project_config = config

            # List all collections for the user to link if desired
            all_collections = await self.qdrant_connector.get_collection_names()
            other_collections = [c for c in all_collections if c != resolved_collection]

            config_json = config.model_dump_json(indent=2)

            lines = [
                f"Project '{project_name}' initialized.",
                f"  Collection: {resolved_collection}",
                "",
                "Please create a `.qdrant-project.json` file in the project root with the following content:",
                "",
                f"```json\n{config_json}\n```",
            ]
            if other_collections:
                lines.append(
                    "\nAvailable collections to link (use qdrant-link): "
                    + ", ".join(other_collections)
                )
            else:
                lines.append("\nNo other collections available to link.")
            return "\n".join(lines)

        async def qdrant_create_collection(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The name of the collection to create")
            ],
        ) -> str:
            """
            Create a new Qdrant collection.
            """
            err = _validate_name(collection_name, "collection name")
            if err:
                return err
            await self.qdrant_connector.ensure_collection_exists(collection_name)
            return f"Collection '{collection_name}' created (or already exists)."

        async def qdrant_list_collections(ctx: Context) -> str:
            """
            List all Qdrant collections with basic info.
            If a project is initialized, marks which collections are linked.
            """
            collection_names = await self.qdrant_connector.get_collection_names()
            if not collection_names:
                return "No collections found."

            lines = ["Available collections:"]
            for name in collection_names:
                info = await self.qdrant_connector.get_collection_info(name)
                points = info.get("points_count", "?") if info else "?"
                status = info.get("status", "?") if info else "?"
                tag = ""
                if self.project_config is not None:
                    if name == self.project_config.collection:
                        tag = " [project]"
                    elif name in self.project_config.linked_collections:
                        tag = " [linked]"
                lines.append(f"  - {name}{tag}: {points} points, status={status}")
            return "\n".join(lines)

        async def qdrant_link(
            ctx: Context,
            collection_name: Annotated[
                str,
                Field(description="The name of the collection to link to the project"),
            ],
        ) -> str:
            """
            Link an existing Qdrant collection to the current project so that
            qdrant-find searches it alongside the project collection.
            """
            err = _validate_name(collection_name, "collection name")
            if err:
                return err
            if self.project_config is None:
                return (
                    "No project initialized. Use qdrant-init-project to create one first."
                )

            # Verify collection exists
            all_collections = await self.qdrant_connector.get_collection_names()
            if collection_name not in all_collections:
                return f"Collection '{collection_name}' does not exist in Qdrant."

            if collection_name == self.project_config.collection:
                return f"Collection '{collection_name}' is already the project collection."

            if collection_name not in self.project_config.linked_collections:
                updated = ProjectConfig(
                    project_name=self.project_config.project_name,
                    collection=self.project_config.collection,
                    linked_collections=self.project_config.linked_collections
                    + [collection_name],
                )
                self.project_config = updated

            config_json = self.project_config.model_dump_json(indent=2)
            return (
                f"Collection '{collection_name}' linked to project '{self.project_config.project_name}'.\n"
                f"Linked collections: {', '.join(self.project_config.linked_collections)}\n\n"
                f"Please update `.qdrant-project.json` in the project root with:\n\n"
                f"```json\n{config_json}\n```"
            )

        async def qdrant_unlink(
            ctx: Context,
            collection_name: Annotated[
                str,
                Field(
                    description="The name of the collection to unlink from the project"
                ),
            ],
        ) -> str:
            """
            Unlink a collection from the current project.
            """
            if self.project_config is None:
                return (
                    "No project initialized. Use qdrant-init-project to create one first."
                )

            if collection_name not in self.project_config.linked_collections:
                return f"Collection '{collection_name}' is not linked to this project."

            new_linked = [
                c
                for c in self.project_config.linked_collections
                if c != collection_name
            ]
            updated = ProjectConfig(
                project_name=self.project_config.project_name,
                collection=self.project_config.collection,
                linked_collections=new_linked,
            )
            self.project_config = updated

            remaining = (
                ", ".join(self.project_config.linked_collections)
                if self.project_config.linked_collections
                else "none"
            )
            config_json = self.project_config.model_dump_json(indent=2)
            return (
                f"Collection '{collection_name}' unlinked from project '{self.project_config.project_name}'.\n"
                f"Remaining linked collections: {remaining}\n\n"
                f"Please update `.qdrant-project.json` in the project root with:\n\n"
                f"```json\n{config_json}\n```"
            )

        async def qdrant_project_info(ctx: Context) -> str:
            """
            Show information about the current Qdrant project configuration.
            """
            if self.project_config is None:
                return "No project initialized. Use qdrant-init-project to create one."

            lines = [
                f"Project: {self.project_config.project_name}",
                f"  Main collection: {self.project_config.collection}",
            ]

            main_info = await self.qdrant_connector.get_collection_info(
                self.project_config.collection
            )
            if main_info:
                lines.append(
                    f"    Points: {main_info.get('points_count', '?')}, "
                    f"Status: {main_info.get('status', '?')}"
                )
            else:
                lines.append("    (collection not found in Qdrant)")

            if self.project_config.linked_collections:
                lines.append("  Linked collections:")
                for cname in self.project_config.linked_collections:
                    info = await self.qdrant_connector.get_collection_info(cname)
                    if info:
                        lines.append(
                            f"    - {cname}: {info.get('points_count', '?')} points, "
                            f"status={info.get('status', '?')}"
                        )
                    else:
                        lines.append(f"    - {cname}: (not found in Qdrant)")
            else:
                lines.append("  Linked collections: none")

            return "\n".join(lines)

        # Register project tools (list-collections and project-info are always available)
        self.tool(
            qdrant_list_collections,
            name="qdrant-list-collections",
            description="List all Qdrant collections. If a project is initialized, shows which are linked.",
        )
        self.tool(
            qdrant_project_info,
            name="qdrant-project-info",
            description="Show information about the current Qdrant project configuration.",
        )

        # Write tools disabled in read-only mode
        if not self.qdrant_settings.read_only:
            self.tool(
                qdrant_init_project,
                name="qdrant-init-project",
                description="Initialize a Qdrant project by creating a .qdrant-project.json config file.",
            )
            self.tool(
                qdrant_create_collection,
                name="qdrant-create-collection",
                description="Create a new Qdrant collection.",
            )
            self.tool(
                qdrant_link,
                name="qdrant-link",
                description="Link an existing Qdrant collection to the current project for multi-collection search.",
            )
            self.tool(
                qdrant_unlink,
                name="qdrant-unlink",
                description="Unlink a collection from the current project.",
            )
