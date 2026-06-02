# Support

The SIL Wheel Agent is a client for a SIL Wheel deployment. For help:

- **Using the agent / SDK**: read `SKILL.md` and the docs in this `knowledge/`
  folder, then open an issue on the project's GitHub repository.
- **Your SIL Wheel server** (auth, data sources, access): contact whoever
  operates the deployment you are pointing `WHEEL_SERVER_URL` at.
- **Bugs / feature requests**: open a GitHub issue with a minimal repro.

This agent talks to whatever server you set in `WHEEL_SERVER_URL`; it ships no
credentials and no hardcoded server. See `.env.template`.
