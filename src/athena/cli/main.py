"""CLI entry point for Athena."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="athena",
        description="Athena - AI-powered runtime debugging agent. "
        "No code changes needed -- just run your script through this tool.",
    )

    # Allow `athena script.py` without a subcommand
    parser.add_argument(
        "script",
        nargs="?",
        help="Python script to debug",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Arguments for the script",
    )

    parser.add_argument(
        "--break-on-entry",
        action="store_true",
        default=False,
        help="Break at the first line of the script (default: false)",
    )
    parser.add_argument(
        "--no-break-on-entry",
        action="store_false",
        dest="break_on_entry",
        help="Don't break at the first line (default)",
    )
    parser.add_argument(
        "--break-on-exception",
        action="store_true",
        default=True,
        help="Break when an unhandled exception occurs",
    )
    parser.add_argument(
        "-b",
        "--break",
        action="append",
        dest="breakpoints",
        metavar="FILE:LINE",
        help="Set initial breakpoint (can be repeated)",
    )
    parser.add_argument(
        "--inject-break",
        action="append",
        dest="injected_breaks",
        metavar="FILE:LINE",
        help="Inject a runtime breakpoint call before FILE:LINE at compile time "
        "(entry script only; can be repeated)",
    )
    parser.add_argument(
        "--focus",
        action="append",
        dest="focus_files",
        metavar="FILE",
        help="Focus debugging on specific files (can be repeated)",
    )
    parser.add_argument(
        "--focus-function",
        action="append",
        dest="focus_functions",
        metavar="FUNC",
        help="Focus debugging on specific functions (can be repeated)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use (default: zai-org/GLM-4.7)",
    )
    parser.add_argument(
        "--trace-memory",
        action="store_true",
        default=False,
        help="Enable tracemalloc from startup",
    )
    parser.add_argument(
        "--restore",
        nargs="?",
        const="latest",
        default=None,
        metavar="STATE",
        help="Restore persisted Athena state (default when flag present: latest)",
    )
    parser.add_argument(
        "--state-dir",
        default=None,
        metavar="PATH",
        help="Directory for persisted Athena state files (default: .athena/state)",
    )
    parser.add_argument(
        "--no-auto-save-state",
        action="store_false",
        dest="auto_save_state",
        default=None,
        help="Disable automatic state snapshots during this run",
    )
    parser.add_argument(
        "--perf-mode",
        action="store_true",
        default=False,
        help="Enable checkpointed performance-debug workflow guidance",
    )

    args = parser.parse_args(argv)

    if args.script is None:
        parser.print_help()
        sys.exit(1)

    if not args.script.endswith(".py"):
        parser.error(f"Expected a .py file, got: {args.script}")

    from athena.cli.runner import ScriptRunner

    runner = ScriptRunner()
    runner.run(
        script_path=args.script,
        script_args=args.script_args,
        break_on_entry=args.break_on_entry,
        break_on_exception=args.break_on_exception,
        initial_breakpoints=args.breakpoints,
        injected_breaks=args.injected_breaks,
        focus_files=args.focus_files,
        focus_functions=args.focus_functions,
        model=args.model,
        trace_memory=args.trace_memory,
        restore_selector=args.restore,
        state_dir=args.state_dir,
        auto_save_state=args.auto_save_state,
        perf_mode=args.perf_mode,
    )


if __name__ == "__main__":
    main()
