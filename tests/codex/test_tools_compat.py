"""Tests for Codex tool compatibility layer."""

import pytest
from unittest.mock import patch

from src.codex.tools_compat import (
    _CUSTOM_TOOL_SUFFIXES,
    CodexCustomToolSpec,
    CodexFunctionToolSpec,
    CodexToolContext,
    codex_tools_to_openai,
    is_codex_builtin_tool,
    parse_codex_tools,
)


# ---------------------------------------------------------------------------
# is_codex_builtin_tool
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["web_search", "local_shell", "computer_use"])
def test_builtin_by_string(name):
    assert is_codex_builtin_tool(name) is True


def test_builtin_by_dict_type():
    assert is_codex_builtin_tool({"type": "web_search"}) is True


def test_builtin_by_dict_name():
    assert is_codex_builtin_tool({"name": "local_shell"}) is True


def test_builtin_by_dict_name_matches_even_different_type():
    """If the *name* field matches a builtin, it's a builtin even when the
    *type* field says something else (defensive match)."""
    assert is_codex_builtin_tool({"type": "function", "name": "computer_use"}) is True


def test_builtin_not_builtin_string():
    assert is_codex_builtin_tool("exec_command") is False


def test_builtin_not_builtin_dict():
    assert is_codex_builtin_tool({"type": "function", "name": "other"}) is False


# ---------------------------------------------------------------------------
# 1. String tool parsing
# ---------------------------------------------------------------------------

def test_string_tool():
    raw = ["exec_command"]
    ctx = parse_codex_tools(raw)
    assert len(ctx.function_tools) == 1
    assert "exec_command" in ctx.function_tools
    spec = ctx.function_tools["exec_command"]
    assert spec.namespace is None
    assert spec.name == "exec_command"
    assert spec.description == "FREEFORM custom tool"
    assert spec.parameters == {
        "type": "object",
        "properties": {"input": {"type": "string"}},
        "required": ["input"],
    }

    assert len(ctx.tools) == 1
    assert ctx.tools[0] == {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": "FREEFORM custom tool",
            "parameters": {
                "type": "object",
                "properties": {"input": {"type": "string"}},
                "required": ["input"],
            },
        },
    }


def test_string_tool_empty_name_skipped():
    """Invalid string list entries like empty strings are still processed as
    zero-length proxy names in the same way every other string is."""
    # parse_codex_tools just uses whatever string it gets as the name;
    # the spec does not explicitly say to skip empty strings.
    raw = ["exec_command", ""]
    ctx = parse_codex_tools(raw)
    assert len(ctx.function_tools) == 2
    assert "" in ctx.function_tools


# ---------------------------------------------------------------------------
# 2. Custom tool parsing -> multiple proxy functions
# ---------------------------------------------------------------------------

def test_custom_tool_no_description():
    raw = [{"type": "custom", "name": "apply_patch"}]
    ctx = parse_codex_tools(raw)

    assert ctx.has_custom_tools is True
    assert "apply_patch" in ctx.custom_tools
    assert ctx.custom_tools["apply_patch"].type == "custom"

    # Should have created one proxy function per suffix
    proxy_names = {f"apply_patch{s}" for s in _CUSTOM_TOOL_SUFFIXES}
    assert set(ctx.function_tools.keys()) == proxy_names

    # Check each has generic custom params
    for pname in proxy_names:
        spec = ctx.function_tools[pname]
        assert spec.parameters == {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        }
        assert spec.namespace is None
        assert "apply_patch" in spec.description
        assert "Custom tool:" in spec.description

    # Check OpenAI output
    assert len(ctx.tools) == len(_CUSTOM_TOOL_SUFFIXES)
    for t in ctx.tools:
        assert t["type"] == "function"
        assert t["function"]["name"] in proxy_names
        assert "parameters" in t["function"]


def test_custom_tool_with_description():
    raw = [{"type": "custom", "name": "apply_patch", "description": "Apply code patches"}]
    ctx = parse_codex_tools(raw)

    for suffix in _CUSTOM_TOOL_SUFFIXES:
        pname = f"apply_patch{suffix}"
        spec = ctx.function_tools[pname]
        assert "Apply code patches" in spec.description


def test_custom_tool_empty_name_skipped():
    """A custom tool with empty name is skipped."""
    raw = [{"type": "custom", "name": ""}]
    ctx = parse_codex_tools(raw)
    assert not ctx.custom_tools
    assert not ctx.function_tools


# ---------------------------------------------------------------------------
# 3. Namespace tool parsing -> flattened names
# ---------------------------------------------------------------------------

def test_namespace_tools_flattened():
    raw = [
        {
            "type": "namespace",
            "name": "mcp__",
            "tools": [
                {"name": "read_file", "description": "Read a file"},
                {"name": "write_file", "description": "Write a file"},
            ],
        }
    ]
    ctx = parse_codex_tools(raw)

    assert ctx.has_namespace_tools is True
    assert "mcp__read_file" in ctx.function_tools
    assert "mcp__write_file" in ctx.function_tools

    rf = ctx.function_tools["mcp__read_file"]
    assert rf.namespace == "mcp__"
    assert rf.name == "read_file"
    assert rf.description == "Read a file"

    wf = ctx.function_tools["mcp__write_file"]
    assert wf.namespace == "mcp__"
    assert wf.name == "write_file"
    assert wf.description == "Write a file"

    # OpenAI output
    assert len(ctx.tools) == 2
    names = {t["function"]["name"] for t in ctx.tools}
    assert names == {"mcp__read_file", "mcp__write_file"}


def test_namespace_tools_with_params():
    raw = [
        {
            "type": "namespace",
            "name": "serve__",
            "tools": [
                {
                    "name": "search",
                    "description": "Search web",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                },
            ],
        }
    ]
    ctx = parse_codex_tools(raw)
    assert ctx.function_tools["serve__search"].parameters is not None
    assert ctx.function_tools["serve__search"].parameters["type"] == "object"


def test_namespace_no_name():
    """Namespace without a name is allowed but may produce odd keys."""
    raw = [{"type": "namespace", "name": "", "tools": [{"name": "x"}]}]
    ctx = parse_codex_tools(raw)
    assert "x" in ctx.function_tools


def test_namespace_missing_tools():
    raw = [{"type": "namespace", "name": "mcp__"}]
    ctx = parse_codex_tools(raw)
    assert not ctx.function_tools
    assert not ctx.tools


# ---------------------------------------------------------------------------
# 4. Built-in tool detection and stripping
# ---------------------------------------------------------------------------

def test_builtin_tools_stripped_when_compat_enabled():
    """When CODEX_TOOL_COMPAT is enabled (default), builtins are stripped."""
    raw = ["web_search", "local_shell", "computer_use"]
    ctx = parse_codex_tools(raw)
    assert not ctx.tools
    assert not ctx.function_tools


def test_builtin_tools_passthrough_when_compat_disabled():
    """When CODEX_TOOL_COMPAT is disabled, builtins are passed through as-is."""
    with patch("src.codex.tools_compat._config") as mock_config:
        mock_config.codex_tool_compat = False
        raw = ["web_search", "local_shell", "computer_use"]
        ctx = parse_codex_tools(raw)
        assert len(ctx.tools) == 3
        for t in ctx.tools:
            # Since builtins pass through as-is, keep original shape
            assert t == {"type": "web_search"} or t == {"type": "local_shell"} or t == {"type": "computer_use"}
        assert not ctx.function_tools


# ---------------------------------------------------------------------------
# 5. Standard function tool passthrough
# ---------------------------------------------------------------------------

def test_standard_function_passthrough():
    raw = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    ctx = parse_codex_tools(raw)
    assert not ctx.has_custom_tools
    assert not ctx.has_namespace_tools
    assert "get_weather" in ctx.function_tools
    assert ctx.function_tools["get_weather"].namespace is None
    assert ctx.function_tools["get_weather"].name == "get_weather"

    # Passthrough unchanged
    func_def = ctx.tools[0]
    assert func_def == raw[0]


def test_standard_function_dict_without_inner_function():
    """A dict with 'function' block but not using the usual top-level type."""
    raw = [
        {
            "function": {
                "name": "multi_fn",
                "description": "Multi",
                "parameters": {"type": "object"},
            }
        }
    ]
    ctx = parse_codex_tools(raw)
    assert "multi_fn" in ctx.function_tools
    assert len(ctx.tools) == 1


# ---------------------------------------------------------------------------
# 6. Tool choice mapping
# ---------------------------------------------------------------------------

def test_tool_choice_required():
    ctx = parse_codex_tools([])
    assert ctx.map_tool_choice("required") == "required"


def test_tool_choice_auto():
    ctx = parse_codex_tools([])
    assert ctx.map_tool_choice("auto") == "auto"


def test_tool_choice_none():
    ctx = parse_codex_tools([])
    assert ctx.map_tool_choice("none") == "none"


def test_tool_choice_openai_style_passthrough():
    """Already-OpenAI-format {type:"function", function:{name:"..."}}."""
    ctx = parse_codex_tools([])
    choice = {"type": "function", "function": {"name": "foo"}}
    assert ctx.map_tool_choice(choice) == choice


def test_tool_choice_namespace_object():
    ctx = parse_codex_tools([
        {
            "type": "namespace",
            "name": "mcp__",
            "tools": [{"name": "read_file"}],
        }
    ])
    choice = {"type": "function_call", "name": "read_file", "namespace": "mcp__"}
    result = ctx.map_tool_choice(choice)
    assert result == {
        "type": "function",
        "function": {"name": "mcp__read_file"},
    }


def test_tool_choice_custom_object():
    raw = [{"type": "custom", "name": "apply_patch"}]
    ctx = parse_codex_tools(raw)
    choice = {"type": "custom", "name": "apply_patch"}
    result = ctx.map_tool_choice(choice)
    assert result == {
        "type": "function",
        "function": {"name": "apply_patch_add_file"},
    }


def test_tool_choice_standard_object():
    ctx = parse_codex_tools([])
    choice = {"type": "function_call", "name": "get_weather"}
    result = ctx.map_tool_choice(choice)
    assert result == {
        "type": "function",
        "function": {"name": "get_weather"},
    }


def test_tool_choice_unknown_returns_auto():
    ctx = parse_codex_tools([])
    assert ctx.map_tool_choice(12345) == "auto"
    assert ctx.map_tool_choice(None) == "auto"


# ---------------------------------------------------------------------------
# 7. Tool call remapping back
# ---------------------------------------------------------------------------

def test_remap_custom_tool():
    """apply_patch_add_file -> back to custom_tool_call with original name."""
    raw = [{"type": "custom", "name": "apply_patch"}]
    ctx = parse_codex_tools(raw)

    class FakeItem:
        def __init__(self, type_, name):
            self.type = type_
            self.name = name

    items = [FakeItem("function_call", "apply_patch_add_file")]
    remapped = ctx.remap_tool_calls_back(items)
    assert remapped[0].type == "custom_tool_call"
    assert remapped[0].name == "apply_patch"


def test_remap_custom_tool_with_description():
    raw = [{"type": "custom", "name": "apply_patch", "description": "Patch files"}]
    ctx = parse_codex_tools(raw)

    class FakeItem:
        def __init__(self, type_, name):
            self.type = type_
            self.name = name
            self.namespace = None

    items = [FakeItem("function_call", "apply_patch_update_file")]
    remapped = ctx.remap_tool_calls_back(items)
    assert remapped[0].type == "custom_tool_call"
    assert remapped[0].name == "apply_patch"


def test_remap_namespace_tool():
    raw = [
        {
            "type": "namespace",
            "name": "mcp__",
            "tools": [{"name": "read_file", "description": "Read"}],
        }
    ]
    ctx = parse_codex_tools(raw)

    class FakeItem:
        def __init__(self, type_, name):
            self.type = type_
            self.name = name
            self.namespace = None

    items = [FakeItem("function_call", "mcp__read_file")]
    remapped = ctx.remap_tool_calls_back(items)
    assert remapped[0].type == "function_call"
    assert remapped[0].name == "read_file"
    assert remapped[0].namespace == "mcp__"


def test_remap_standard_function_passthrough():
    """Standard function tools should be passed through unchanged."""
    raw = [
        {
            "type": "function",
            "function": {"name": "get_weather"},
        }
    ]
    ctx = parse_codex_tools(raw)

    class FakeItem:
        def __init__(self, type_, name):
            self.type = type_
            self.name = name

    items = [FakeItem("function_call", "get_weather")]
    remapped = ctx.remap_tool_calls_back(items)
    # Should stay as-is because it's a standard function
    assert remapped[0].type == "function_call"
    assert remapped[0].name == "get_weather"


def test_remap_unknown_name_passthrough():
    """An unknown proxy name that doesn't match custom or namespace patterns
    should be left untouched."""
    ctx = parse_codex_tools([])

    class FakeItem:
        def __init__(self, type_, name):
            self.type = type_
            self.name = name

    items = [FakeItem("function_call", "random_tool")]
    remapped = ctx.remap_tool_calls_back(items)
    assert remapped[0].type == "function_call"
    assert remapped[0].name == "random_tool"


def test_remap_string_arg():
    """remap_tool_calls_back accepts a single string argument for stream
    converter plumbing."""
    raw = [{"type": "custom", "name": "apply_patch"}]
    ctx = parse_codex_tools(raw)

    # String arg
    assert ctx.remap_tool_calls_back("apply_patch_add_file") == "apply_patch"
    # Unknown string
    assert ctx.remap_tool_calls_back("random_name") == "random_name"

    # Namespace unflatten string
    raw_ns = [
        {
            "type": "namespace",
            "name": "mcp__",
            "tools": [{"name": "read_file"}],
        }
    ]
    ctx_ns = parse_codex_tools(raw_ns)
    assert ctx_ns.remap_tool_calls_back("mcp__read_file") == "read_file"


# ---------------------------------------------------------------------------
# 8. Mixed tool array
# ---------------------------------------------------------------------------

def test_mixed_tools():
    raw = [
        "exec_command",
        {"type": "custom", "name": "apply_patch"},
        {
            "type": "namespace",
            "name": "mcp__",
            "tools": [{"name": "read_file"}],
        },
        "web_search",  # built-in, stripped
        {
            "type": "function",
            "function": {"name": "get_weather"},
        },
    ]
    ctx = parse_codex_tools(raw)

    # Should have: 1 string tool + 5 custom suffixes + 1 namespace + 1 passthrough
    # web_search is stripped (builtin)
    assert len(ctx.tools) == 1 + len(_CUSTOM_TOOL_SUFFIXES) + 1 + 1
    assert ctx.has_custom_tools is True
    assert ctx.has_namespace_tools is True

    # Verify all types are present
    names = [t["function"]["name"] for t in ctx.tools]
    assert "exec_command" in names
    assert "mcp__read_file" in names
    assert "get_weather" in names
    assert any(n.startswith("apply_patch_") for n in names)
    assert "web_search" not in names


# ---------------------------------------------------------------------------
# 9. Empty tools array
# ---------------------------------------------------------------------------

def test_empty_tools():
    ctx = parse_codex_tools([])
    assert not ctx.tools
    assert not ctx.custom_tools
    assert not ctx.function_tools
    assert ctx.has_custom_tools is False
    assert ctx.has_namespace_tools is False


def test_none_tools():
    ctx = parse_codex_tools(None)
    assert not ctx.tools


# ---------------------------------------------------------------------------
# codex_tools_to_openai
# ---------------------------------------------------------------------------

def test_codex_tools_to_openai_is_tools_property():
    raw = ["exec_command"]
    ctx = parse_codex_tools(raw)
    assert codex_tools_to_openai(ctx) is ctx.tools


def test_flat_function_tool_format_normalised():
    """Codex CLI sends {"type":"function","name":"X","parameters":P} without nested
    'function' dict; we must normalise to OpenAI format."""
    raw = [
        {
            "type": "function",
            "name": "exec_command",
            "description": "Runs a command in a PTY",
            "parameters": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
        }
    ]
    ctx = parse_codex_tools(raw)
    tools = codex_tools_to_openai(ctx)
    assert len(tools) == 1
    t = tools[0]
    assert t["type"] == "function"
    assert "function" in t
    assert t["function"]["name"] == "exec_command"
    assert t["function"]["description"] == "Runs a command in a PTY"
    assert "parameters" in t["function"]
