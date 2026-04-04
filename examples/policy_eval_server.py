"""Start an rpyc server that exposes PolicyEvaluator remotely.

Clients connect and call ``from_config`` / ``from_yaml`` to initialize,
then drive evaluation via ``reset`` / ``update`` / ``get_observation``.

Usage::

    python examples/policy_eval_server.py
    python examples/policy_eval_server.py --host 0.0.0.0 --port 9999
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="PolicyEvaluator rpyc server")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=18861)
    args = parser.parse_args()

    from auto_atom.ipc import serve_policy_evaluator

    serve_policy_evaluator(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
