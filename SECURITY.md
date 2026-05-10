# Security Policy

## Supported Versions

Security patches are released for the latest version only.

| Version | Supported |
|---|---|
| 1.0.x | ✅ Yes |
| < 1.0 | ❌ No |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email **security@aiexponent.com** with:

1. Description of the vulnerability
2. Steps to reproduce
3. Potential impact
4. Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide an assessment within 5 business days.

## Disclosure Policy

We follow coordinated disclosure. Please:

- Allow us reasonable time to fix the issue before public disclosure
- Do not exploit the vulnerability beyond what is needed to demonstrate it
- Do not access or modify other users' data

We credit reporters in release notes unless they prefer to remain anonymous.

## Data Handling

This tool processes RAG evaluation data locally. When using the server:

- Evaluation samples and results are stored in a local SQLite database (`eval_results.db` at the project root by default; gitignored). The database is created on first run and never sent off-host by the harness itself.
- LLM judge calls are made to your configured provider (Gemini or OpenAI) — review their privacy policies.
- No data is sent to AI Exponent LLC servers.

## Contact

security@aiexponent.com
