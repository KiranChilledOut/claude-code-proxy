"""Server-side execution of web-search tools (Tavily-backed).

Claude Code's ``WebSearch`` (and the Anthropic ``web_search`` server tool) cannot
run behind a non-Anthropic backend — there is no search engine on the other side,
so they resolve to "0 searches". When ``TAVILY_API_KEY`` is configured, the proxy
executes the search itself and feeds the results back to the model in a bounded
loop, returning the final answer. The web tools therefore run invisibly,
mirroring how Anthropic's server tools behave.

When ``TAVILY_API_KEY`` is unset (or ``SERVER_SEARCH_ENABLED=false``) this module
is inert and the proxy behaves exactly as before.
"""

from typing import Any, Dict, List

import httpx

from src.core.config import config
from src.core.logging import logger

# Tool names (case-insensitive) the proxy executes itself.
_SEARCH_NAMES = {"web_search", "websearch"}

# Schema we force on search tools when forwarding, so the backend model emits a
# clean {"query": ...} call instead of a no-arg / control-token blob.
SEARCH_TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The search query."}
    },
    "required": ["query"],
}


SEARCH_TOOL_SYSTEM_SUPPLEMENT = (
    "Web search note: this environment executes web search server-side. When you "
    "need to search the web, call the web search tool (web_search / WebSearch) by "
    "ITSELF in its own turn — do not batch it together with other tool calls in the "
    "same response. The search runs and its results are returned to you before you "
    "continue. (Batching it with other tools prevents the search from executing.)"
)


def is_search_tool(tool: Any) -> bool:
    """True if a tool definition is a web-search tool (by name or server type)."""
    name = (getattr(tool, "name", "") or "").lower()
    ttype = (getattr(tool, "type", "") or "").lower()
    return name in _SEARCH_NAMES or ttype.startswith("web_search")


def is_search_tool_name(name: str) -> bool:
    return (name or "").lower() in _SEARCH_NAMES


def request_has_search_tool(claude_request) -> bool:
    """True if search execution is enabled and the request offers a search tool."""
    if not config.tavily_api_key or not config.server_search_enabled:
        return False
    for tool in getattr(claude_request, "tools", None) or []:
        if is_search_tool(tool):
            return True
    return False


def _tc_name(tc: Dict[str, Any]) -> str:
    return (tc.get("function") or {}).get("name") or ""


async def tavily_search(query: str) -> str:
    """Run a Tavily search and return a compact, model-readable result string."""
    if not query:
        return "No search query was provided."
    if not config.tavily_api_key:
        return "Web search is not configured on this proxy."

    payload = {
        "query": query,
        "max_results": config.tavily_max_results,
        "include_answer": True,
        "search_depth": "basic",
    }
    headers = {"Authorization": f"Bearer {config.tavily_api_key}"}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.post(
                "https://api.tavily.com/search", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as exc:  # network / auth / parse — never raise into the loop
        logger.warning(f"[server_tools] Tavily search failed: {exc}")
        return f"Web search error: {exc}"

    lines: List[str] = []
    if data.get("answer"):
        lines.append(f"Answer: {data['answer']}")
    for item in (data.get("results") or [])[: config.tavily_max_results]:
        title = item.get("title", "")
        url = item.get("url", "")
        content = item.get("content", "")
        lines.append(f"- {title} ({url})\n  {content}")
    return "\n".join(lines) if lines else "No results found."


async def run_search_loop(openai_request: Dict[str, Any], openai_client, request_id):
    """Run the backend, executing owned search tool calls server-side, until the
    model produces an answer (or a non-search tool call the client must handle).

    Returns a final OpenAI response dict. Any pending tool calls in it are NOT
    owned search tools, so the caller's normal conversion/passthrough applies.
    """
    # Imported lazily to keep module import order simple (response_converter
    # imports request_converter which imports this module's siblings).
    from src.conversion.response_converter import _finalize_tool_args

    req = dict(openai_request)
    req["stream"] = False
    messages = list(req.get("messages", []))
    response: Dict[str, Any] = {}

    for _ in range(max(1, config.server_search_max_iters)):
        req["messages"] = messages
        response = await openai_client.create_chat_completion(req, request_id)

        choice = (response.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            return response

        search_calls = [tc for tc in tool_calls if is_search_tool_name(_tc_name(tc))]
        # Nothing to execute, or a mix with client tools we cannot run here:
        # hand the whole response back for normal handling.
        if not search_calls or len(search_calls) != len(tool_calls):
            return response

        # Record the assistant turn, then execute each search and append results.
        messages.append(
            {
                "role": "assistant",
                "content": msg.get("content"),
                "tool_calls": tool_calls,
            }
        )
        for tc in search_calls:
            raw_args = (tc.get("function") or {}).get("arguments") or "{}"
            _, _, parsed = _finalize_tool_args(_tc_name(tc), raw_args)
            parsed = parsed or {}
            query = parsed.get("query") or parsed.get("q") or ""
            result = await tavily_search(query)
            logger.info(f"[server_tools] executed web_search query={query!r}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "content": result,
                }
            )

    # Max iterations reached while still searching: force a final answer by
    # dropping the search tools so the model must respond with text.
    req["messages"] = messages
    req.pop("tools", None)
    req.pop("tool_choice", None)
    response = await openai_client.create_chat_completion(req, request_id)
    return response
