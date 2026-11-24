"""HTTP routes, one module per resource."""
from . import collections, evaluation, files, health, jobs, search, uploads

ROUTERS = [health.router, collections.router, search.router, files.router, uploads.router, jobs.router,
           evaluation.router]

__all__ = ["ROUTERS"]
