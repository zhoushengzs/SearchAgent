"""Evaluate the Base model with the ordinary prompt and six native tools.

The prompt override is scoped to the worker process.
"""
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path

from evaluation import unified_run
from evaluation.prepare import digest
from evaluation.run import SYSTEM


def extension_identity():
    root = Path(__file__).resolve().parents[1]
    return {'evaluation/' + name: digest(root / 'evaluation' / name) for name in
            ['plain6_run.py', 'plain6_benchmark.py', 'plain6_benchmark.sh', 'plain6_score.sh']}


def ordinary_system(session):
    # Combine the ordinary instructions with the native tool schemas.
    schemas = session.get_all_tools().values()
    return SYSTEM + '\n\n# Tools\n<tools>\n' + '\n'.join(
        json.dumps(schema, ensure_ascii=False) for schema in schemas) + '\n</tools>'


@contextmanager
def ordinary_prompt(agent, session):
    prompt = ordinary_system(session)
    saved = [(session, 'system_prompt', session.system_prompt),
             (agent, 'system_prompt', agent.system_prompt),
             (agent, 'DEEPRESEARCH_SYSTEM_PROMPT', agent.DEEPRESEARCH_SYSTEM_PROMPT)]
    session.system_prompt = agent.system_prompt = lambda: prompt
    agent.DEEPRESEARCH_SYSTEM_PROMPT = prompt
    try:
        yield prompt
    finally:
        for module, name, original in saved:
            setattr(module, name, original)


def main():
    original_load = unified_run.load_runtime
    original_identity = unified_run.code_identity
    with ExitStack() as stack:
        def load(directory):
            agent, session = original_load(directory)
            stack.enter_context(ordinary_prompt(agent, session))
            return agent, session
        unified_run.load_runtime = load
        unified_run.code_identity = lambda: {**original_identity(), **extension_identity()}
        try:
            unified_run.main()
        finally:
            unified_run.load_runtime = original_load
            unified_run.code_identity = original_identity


if __name__ == '__main__':
    main()
