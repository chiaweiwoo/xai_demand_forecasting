# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**xai_demand_forecasting** — Explainable AI demand forecasting project. Greenfield as of 2026-06-26.

## Stack

- **Backend / ML:** Python, managed with `uv`
- **Frontend (if added):** React + TypeScript, Chakra UI, Vite
- **Database:** Supabase (Postgres) — use Supabase client only, never direct DB connection
- **Deployment:** Vercel (frontend)

## Commands

> Populate these as the project is set up.

```bash
# Install dependencies (Python)
uv sync

# Run tests
uv run pytest

# Run a single test
uv run pytest path/to/test_file.py::test_name
```

## Architecture

> Document the architecture here as it is built out.
