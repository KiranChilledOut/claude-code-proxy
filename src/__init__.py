"""Claude Code Proxy

A proxy server that enables Claude Code to work with OpenAI-compatible API providers.
"""

from dotenv import load_dotenv

# Load environment variables from .env file (env vars already exported in the
# shell take precedence unless we deliberately override them).  The proxy is
# intended to defer to .env as its source of truth, so we use override=True.
load_dotenv(override=True)
__version__ = "1.0.0"
__author__ = "Claude Code Proxy"
