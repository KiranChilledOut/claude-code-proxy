"""Textual TUI entry point: install.sh bootstrap calls this."""

from __future__ import annotations

from textual.app import App

from scripts.install_ui.screens import WelcomeScreen
from scripts.install_ui.utils import InstallState


class InstallerApp(App[None]):
    """The full-screen TUI wizard."""

    TITLE = "Claude Code Proxy"
    SUB_TITLE = "Setup Wizard"
    ENABLE_COMMAND_PALETTE = False

    # Shared state, mutated in-place by screens as the user progresses.
    state = InstallState()

    def on_mount(self) -> None:
        self.push_screen(WelcomeScreen())


# CSS is loaded from the sibling styles.css file relative to this module.
if __name__ == "__main__":
    InstallerApp(css_path="scripts/install_ui/styles.css").run()
