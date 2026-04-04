"""Remote PolicyEvaluator via rpyc — isolated from the main auto_atom imports.

Server side (requires simulation deps)::

    from auto_atom.ipc import serve_policy_evaluator
    serve_policy_evaluator(host="0.0.0.0", port=18861)

Client side (no simulation deps needed)::

    from auto_atom.ipc import RemotePolicyEvaluator
    evaluator = RemotePolicyEvaluator("localhost", 18861)
"""

from .client import RemotePolicyEvaluator
from .service import build_server_config, serve_policy_evaluator

__all__ = [
    "RemotePolicyEvaluator",
    "build_server_config",
    "serve_policy_evaluator",
]
