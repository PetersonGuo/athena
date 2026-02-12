"""athena - AI-powered runtime debugging agent.

No code changes needed! Run your script through the debugger:

    $ athena script.py

Or if you prefer the import approach:

    from athena import debug
    debug()  # drops into AI debugger at this point

Or use as PYTHONBREAKPOINT:

    $ PYTHONBREAKPOINT=athena.breakpoint python script.py
"""

from __future__ import annotations

__version__ = "0.1.0"

# Lazy session for import-based API
_session = None


def debug(*, model: str | None = None, **kwargs) -> None:
    """Drop into the AI debugging REPL at the call site.

    Equivalent to pdb.set_trace() but with an AI debugger.
    Pauses execution and starts an interactive session.

    This is the optional import-based API. The primary interface
    is the CLI: `athena script.py`
    """
    import sys

    from athena.repl.session import DebugSession
    from athena.utils.config import DebugConfig

    global _session
    if _session is None:
        config = DebugConfig.from_env()
        if model:
            config.model = model
        _session = DebugSession(config=config)

    frame = sys._getframe().f_back
    _session.debugger.set_trace(frame)


def breakpoint(**kwargs) -> None:
    """Alias for debug(). Works with PYTHONBREAKPOINT env var."""
    import sys

    from athena.repl.session import DebugSession
    from athena.utils.config import DebugConfig

    global _session
    if _session is None:
        config = DebugConfig.from_env()
        _session = DebugSession(config=config)

    frame = sys._getframe().f_back
    _session.debugger.set_trace(frame)
