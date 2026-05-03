import os
import sys

# Make `from src.* import ...` work when running `pytest` from the repo root
# without an editable install. Several existing test files (test_image_routing,
# test_observability, test_client_key_policy, test_computer_use) rely on this
# instead of carrying their own per-file sys.path hack.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Keep unit tests self-contained; production keys are not required.
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-anthropic-key")
