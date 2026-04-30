# License Apache 2.0: (c) 2026 Athena-Reply

"""Run the open-arena evaluation sweep.

Thin wrapper around `src.cli:main` so the sweep can be launched with
`python evaluate.py`. Equivalent to the `arena` console script installed
by `uv sync`; `arena -c config.yaml` and `python evaluate.py -c config.yaml`
do the same thing.
"""

from src.cli import main

if __name__ == "__main__":
    main()
