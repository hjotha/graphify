## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, use the Graphify skill with `--update` (runs `detect_incremental`, re-extracting only new/changed files); do not run the bare `graphify update <path>` CLI (it re-parses the whole corpus).
