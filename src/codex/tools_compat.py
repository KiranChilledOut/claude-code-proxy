"""Codex-specific tool format compatibility layer.

Handles conversion between Codex Responses API tool conventions and the
OpenAI function tool format consumed by our upstream backends.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from src.core.config import config as _config
except Exception:  # pragma: no cover
    _config = None


# ---------------------------------------------------------------------------
# Spec records
# ---------------------------------------------------------------------------

class CodexCustomToolSpec:
    """Record for a parsed Codex custom tool."""

    def __init__(self, name: str, tool_type: str = "custom", description: str = ""):
        self.name = name
        self.type = tool_type
        self.description = description


class CodexFunctionToolSpec:
    """Record for a parsed function / namespace / string tool.

    ``namespace`` is ``None`` for plain function or string tools.
    """

    def __init__(
        self,
        namespace: Optional[str],
        name: str,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.namespace = namespace
        self.name = name
        self.description = description
        self.parameters = parameters


# ---------------------------------------------------------------------------
# Codex CLI tool types that are NOT built-ins but should be converted to
# function tools.  The Codex CLI sends these with type keys like
# "text_editor", "exec_command", "apply_patch", etc.  The upstream Nebius
# backend only understands OpenAI-style function tools, so we repackage them.
# ---------------------------------------------------------------------------
_KNOWN_CODEX_TOOL_TYPES = {
    "text_editor",
    "exec_command",
    "shell",
    "bash",
    "cat_files",
    "glob_files",
    "apply_patch",
    "web_search",
    "computer_use",
}

BUILTIN_TOOLS = {"web_search", "local_shell", "computer_use"}

_CUSTOM_TOOL_SUFFIXES = (
    "_add_file",
    "_update_file",
    "_delete_file",
    "_undo_edit",
    "_batch",
)

_GENERIC_STRING_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
}

_GENERIC_CUSTOM_PARAMS: Dict[str, Any] = {
    "type": "object",
    "properties": {"input": {"type": "string"}},
    "required": ["input"],
}


# ---------------------------------------------------------------------------
# CodexToolContext
# ---------------------------------------------------------------------------

@dataclass
class CodexToolContext:
    """Parsed representation of a Codex ``tools`` array.

    ``custom_tools`` maps the *original* custom tool name to its spec.
    ``function_tools`` maps the *proxy / flattened* name to the original spec.

    The ``tools`` property returns the complete list of OpenAI-format function
    tool dicts that should be forwarded to the upstream backend.
    """

    custom_tools: Dict[str, CodexCustomToolSpec] = field(default_factory=dict)
    function_tools: Dict[str, CodexFunctionToolSpec] = field(default_factory=dict)
    has_custom_tools: bool = False
    has_namespace_tools: bool = False
    has_search_tool: bool = False
    _tools: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    @property
    def tools(self) -> List[Dict[str, Any]]:
        return self._tools

    # ------------------------------------------------------------------
    # Helper used by request_converter / stream_converter
    # ------------------------------------------------------------------

    def map_tool_choice(self, tool_choice: Any) -> Any:
        """Map a Codex ``tool_choice`` value to an OpenAI-compatible one."""
        if tool_choice == "required":
            return "required"
        if tool_choice in ("auto", "none"):
            return tool_choice

        if isinstance(tool_choice, dict):
            # Already OpenAI-style — passthrough
            if (
                tool_choice.get("type") == "function"
                and "function" in tool_choice
            ):
                return tool_choice

            name = tool_choice.get("name", "")
            namespace = tool_choice.get("namespace")

            if namespace:
                return {
                    "type": "function",
                    "function": {"name": f"{namespace}{name}"},
                }

            if name in self.custom_tools:
                return {
                    "type": "function",
                    "function": {"name": f"{name}{_CUSTOM_TOOL_SUFFIXES[0]}"},
                }

            # Standard function
            return {
                "type": "function",
                "function": {"name": name},
            }

        return "auto"

    def _remap_name(self, proxy_name: str) -> str:
        """Remap a single proxy function name back to its original Codex name.

        Used by the stream converter for on-the-fly name resolution.
        """
        # Custom tools: strip known suffix
        for suffix in _CUSTOM_TOOL_SUFFIXES:
            if proxy_name.endswith(suffix):
                base = proxy_name[: -len(suffix)]
                if base in self.custom_tools:
                    return base
        # Namespace tools: unflatten
        for key, spec in self.function_tools.items():
            if spec.namespace and proxy_name == key:
                return spec.name
        return proxy_name

    def remap_tool_calls_back(self, tool_calls: Any) -> Any:
        """Remap proxy function names back to original Codex conventions.

        Accepts either a ``str`` (stream-converter plumbing) or a list of
        objects with ``type`` / ``name`` / ``namespace`` attributes
        (response-converter pipeline).
        """
        if isinstance(tool_calls, str):
            return self._remap_name(tool_calls)

        for item in tool_calls:
            item_type = getattr(item, "type", None)
            if item_type != "function_call":
                continue

            name = getattr(item, "name", None) or ""

            # ---- Custom tools ----
            for suffix in _CUSTOM_TOOL_SUFFIXES:
                if name.endswith(suffix):
                    base = name[: -len(suffix)]
                    if base in self.custom_tools:
                        item.type = "custom_tool_call"
                        item.name = base
                        break

            # ---- Namespace tools (only touch real namespace proxies) ----
            if name in self.function_tools:
                spec = self.function_tools[name]
                if spec.namespace:
                    item.namespace = spec.namespace
                    item.name = spec.name

            # Standard function tools: passthrough (name unchanged)

        return tool_calls


# ---------------------------------------------------------------------------
# Public standalone helpers
# ---------------------------------------------------------------------------

def is_codex_builtin_tool(tool: Any) -> bool:
    """Return ``True`` if *tool* is one of Codex's built-in tools."""
    if isinstance(tool, str):
        return tool in BUILTIN_TOOLS
    if isinstance(tool, dict):
        return (
            tool.get("type", "") in BUILTIN_TOOLS
            or tool.get("name", "") in BUILTIN_TOOLS
        )
    return False


def parse_codex_tools(raw_tools: List[Any]) -> CodexToolContext:
    """Parse a Codex ``tools`` array into a :class:`CodexToolContext`.

    Handles string tools, custom tools, namespace tools, built-in tools, and
    standard ``type: "function"`` dicts.
    """
    ctx = CodexToolContext()
    if not raw_tools:
        return ctx

    if _config is not None:
        try:
            strip_builtins = _config.codex_tool_compat
        except AttributeError:
            strip_builtins = True
    else:
        strip_builtins = True

    for tool in raw_tools:
        # --- Built-ins (stripped or promoted) ---
        if is_codex_builtin_tool(tool):
            # When Tavily is configured, promote web_search to a function tool
            # instead of stripping it, so the proxy can execute it server-side.
            if _config is not None:
                has_tavily = getattr(_config, "tavily_api_key", "") and getattr(
                    _config, "server_search_enabled", True
                )
            else:
                has_tavily = False
            if has_tavily:
                if isinstance(tool, str):
                    name = tool
                else:
                    name = tool.get("type") or tool.get("name") or ""
                if name == "web_search":
                    ctx._tools.append({
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "description": "Search the web for current information.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string", "description": "The search query."}
                                },
                                "required": ["query"],
                            },
                        },
                    })
                    ctx.function_tools["web_search"] = CodexFunctionToolSpec(
                        namespace=None,
                        name="web_search",
                        description="Search the web for current information.",
                        parameters={
                            "type": "object",
                            "properties": {
                                "query": {"type": "string", "description": "The search query."}
                            },
                            "required": ["query"],
                        },
                    )
                    ctx.has_search_tool = True
                    continue
            if not strip_builtins:
                ctx._tools.append(tool if isinstance(tool, dict) else {"type": tool})
            continue

        # --- String tools ---
        if isinstance(tool, str):
            proxy_name = tool
            ctx.function_tools[proxy_name] = CodexFunctionToolSpec(
                namespace=None,
                name=proxy_name,
                description="FREEFORM custom tool",
                parameters=_GENERIC_STRING_PARAMS,
            )
            ctx._tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": proxy_name,
                        "description": "FREEFORM custom tool",
                        "parameters": _GENERIC_STRING_PARAMS,
                    },
                }
            )
            continue

        if not isinstance(tool, dict):
            continue

        tool_type = tool.get("type", "")

        # --- Namespace tools ---
        if tool_type == "namespace":
            namespace = tool.get("name", "")
            nested_tools = tool.get("tools", [])
            ctx.has_namespace_tools = True
            for nt in nested_tools:
                if not isinstance(nt, dict):
                    continue
                original_name = nt.get("name", "")
                if not original_name:
                    continue
                proxy_name = f"{namespace}{original_name}"
                desc = nt.get("description") or ""
                params = nt.get("parameters")
                ctx.function_tools[proxy_name] = CodexFunctionToolSpec(
                    namespace=namespace,
                    name=original_name,
                    description=desc,
                    parameters=params,
                )
                func_def: Dict[str, Any] = {
                    "type": "function",
                    "function": {
                        "name": proxy_name,
                        "description": desc,
                        "parameters": params,
                    },
                }
                if not desc:
                    del func_def["function"]["description"]
                if not params:
                    del func_def["function"]["parameters"]
                ctx._tools.append(func_def)
            continue

        # --- Custom tools ---
        if tool_type == "custom":
            name = tool.get("name", "")
            if not name:
                continue
            desc = tool.get("description", "")
            ctx.custom_tools[name] = CodexCustomToolSpec(
                name=name,
                tool_type="custom",
                description=desc,
            )
            ctx.has_custom_tools = True
            for suffix in _CUSTOM_TOOL_SUFFIXES:
                proxy_name = f"{name}{suffix}"
                op_desc = f"Custom tool: {name} ({suffix[1:]})"
                if desc:
                    op_desc = f"{desc} ({suffix[1:]})"
                ctx.function_tools[proxy_name] = CodexFunctionToolSpec(
                    namespace=None,
                    name=proxy_name,
                    description=op_desc,
                    parameters=_GENERIC_CUSTOM_PARAMS,
                )
                ctx._tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": proxy_name,
                            "description": op_desc,
                            "parameters": _GENERIC_CUSTOM_PARAMS,
                        },
                    }
                )
            continue

        # --- Codex CLI tool types (text_editor, exec_command, etc.) ---
        # These are not built-ins (not in BUILTIN_TOOLS) and the Codex CLI sends
        # them with their own type key.  Nebius only supports function tools, so
        # we convert them to equivalent OpenAI-format function tools.
        if tool_type in _KNOWN_CODEX_TOOL_TYPES:
            name = tool.get("name") or tool_type
            desc = tool.get("description", f"Codex tool: {tool_type}")
            params = tool.get("parameters") or _GENERIC_STRING_PARAMS
            ctx.function_tools[name] = CodexFunctionToolSpec(
                namespace=None, name=name, description=desc, parameters=params
            )
            ctx._tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": params,
                    },
                }
            )
            continue

        # --- Unknown tool types (image_generation, etc.) — strip them ---
        if tool_type not in ("function", "namespace", "custom", ""):
            if not strip_builtins:
                ctx._tools.append(tool)
            continue

        # --- Standard function tools (passthrough with Codex flat-format normalisation) ---
        # Codex CLI 0.130+ sends flat tools:
        #   {"type":"function","name":"X","description":"...","parameters":{...}}
        # OpenAI format requires nested "function" dict.
        if tool_type == "function" or "function" in tool:
            func_block = tool.get("function", {})
            if not func_block and tool.get("name"):
                # Flat format — normalise to nested
                func_block = {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                }
                if "parameters" in tool:
                    func_block["parameters"] = tool["parameters"]
                tool = {"type": "function", "function": func_block}
            elif isinstance(func_block, dict):
                func_name = func_block.get("name", "")
                if func_name:
                    ctx.function_tools[func_name] = CodexFunctionToolSpec(
                        namespace=None,
                        name=func_name,
                        description=func_block.get("description") or None,
                        parameters=func_block.get("parameters") or None,
                    )
            ctx._tools.append(tool)
            continue

        # Unrecognised — include as-is (defensive)
        ctx._tools.append(tool)

    return ctx


def codex_tools_to_openai(ctx: CodexToolContext) -> List[Dict[str, Any]]:
    """Return the OpenAI-format tool list stored in *ctx*."""
    return ctx.tools
