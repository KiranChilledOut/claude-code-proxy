from types import SimpleNamespace

from src.conversion.computer_use import convert_schema_less_tools


def test_convert_schema_less_tools_converts_supported_tools_and_builds_prompt():
    tools = [
        SimpleNamespace(
            name="computer",
            type="computer_20251124",
            display_width_px=1440,
            display_height_px=900,
        ),
        SimpleNamespace(name="bash", type="bash_20250124"),
        SimpleNamespace(name="str_replace_based_edit_tool", type="text_editor_20250728"),
        SimpleNamespace(name="lookup", type="custom_tool"),
    ]

    converted, supplement, has_computer_use = convert_schema_less_tools(tools)

    assert has_computer_use is True
    assert len(converted) == 4

    computer_tool = converted[0]
    assert computer_tool["type"] == "function"
    assert computer_tool["function"]["name"] == "computer"
    assert "1440x900" in computer_tool["function"]["description"]
    assert "open_app" in computer_tool["function"]["parameters"]["properties"]["action"]["enum"]
    assert "open_url" in computer_tool["function"]["parameters"]["properties"]["action"]["enum"]
    assert (
        "cursor_position" in computer_tool["function"]["parameters"]["properties"]["action"]["enum"]
    )

    bash_tool = converted[1]
    assert bash_tool["function"]["name"] == "bash"
    assert bash_tool["function"]["parameters"]["properties"]["command"]["type"] == "string"

    editor_tool = converted[2]
    assert editor_tool["function"]["name"] == "str_replace_based_edit_tool"
    assert editor_tool["function"]["parameters"]["required"] == ["command", "path"]

    assert converted[3] is None

    assert "[Computer Use Environment]" in supplement
    assert "open_app" in supplement
    assert "open_url" in supplement
    assert "1440x900" in supplement
    assert "text editor tool named 'str_replace_based_edit_tool'" in supplement
