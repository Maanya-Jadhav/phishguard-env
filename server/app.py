"""
server/app.py – OpenEnv Validator Entry Point
=============================================

The OpenEnv validator expects the FastAPI application at `server.app:app`
and starts it with:
    uvicorn server.app:app --host 0.0.0.0 --port 7860

This file re-exports the fully configured `app` object from `env.py`
so all routes, middleware, and the environment state remain in one place.
No logic is duplicated here.
"""

import sys
import os

# Allow `import env` to resolve from the project root when this module
# is loaded as `server.app` (i.e. when the working directory is /app
# but Python's module path may not include /app automatically).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import app  # re-export — the validator binds to this name

__all__ = ["app"]


def main():
    """Entry point for `[project.scripts]` and direct execution."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
