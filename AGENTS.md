# Repository guidance

- Treat `~/physicar-ai-sim-docker` as an external, read-only dependency unless the user explicitly authorizes changes.
- Keep raw rosbags, image datasets, checkpoints, generated models, and large logs out of Git.
- Keep source, tests, configuration, documentation, and compact curated metrics in Git.
- Keep the canonical baseline distinct from experimental configurations and results.
- Never present simulator success as real-robot success.
- Do not commit or push unless the user explicitly authorizes it.
