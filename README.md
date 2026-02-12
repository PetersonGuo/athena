# Description:
A runtime debugging agent which inspects the call stack and program state to verify things like memory issues and crashes

# Next steps:
Restorable state to allow agent to quickly resume good for long pieces of code or variable states

# Restorable State (Best-Effort Resume)

Athena now supports persistent state snapshots in `.athena/state/` so you can
resume long debugging workflows without exact process checkpointing.

## What gets saved
- Breakpoints (with conditions and context snippets)
- Watches
- Focus filters
- Runtime summaries (last stop location, stack, safe locals summary)
- Memory summary metadata (snapshot labels and traced memory summary)
- Compact agent conversation summary

## What does not get saved
- Live frame/object references
- Full heap/object graphs
- Secrets/API keys
- Full unbounded transcript

## CLI usage
```bash
# Restore latest compatible state
athena examples/simple_bug.py --restore

# Restore a named/manual state
athena examples/simple_bug.py --restore checkpoint-1

# Custom state directory
athena examples/simple_bug.py --state-dir .athena/state

# Disable auto-save snapshots
athena examples/simple_bug.py --no-auto-save-state

# Enable performance workflow steering
athena examples/simple_bug.py --perf-mode
```

## REPL commands
- `/save-state [NAME]`
- `/load-state [latest|NAME|PATH]`
- `/list-states`
- `/restore-next [latest|NAME|PATH]`
- `/perf-checkpoint [LABEL]`
- `/perf-checkpoints`
- `/perf-compare LABEL_A LABEL_B`
- `/perf-issue [TITLE]`

Restore is **best-effort**: Athena reapplies controls and remaps breakpoints by
saved code snippets when code has drifted, with warnings for anything skipped.

Performance debugging can combine checkpoint state restore with runtime perf
checkpoints (wall/process/memory deltas). The agent can queue restore
automatically for the next rerun when immediate apply is not appropriate.

Athena now also takes best-effort rolling checkpoints before state-changing
debugger actions (file edits, breakpoint/watch/focus changes), so non-perf
debug workflows can restore quickly as well.
