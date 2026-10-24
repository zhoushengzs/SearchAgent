"""Download exact upstream suite versions, without copying datasets or secrets."""
import argparse
from pathlib import Path
import subprocess

REPOSITORIES = {
    'AgentVista': ('https://github.com/hkust-nlp/AgentVista.git', 'a3385d51d0c3dd0c8501350a08d52065e3e94ab4'),
    'Agent-R1': ('https://github.com/0russwest0/Agent-R1.git', 'b124aa46534cbf2fb8bc8af11405774984c42ac7'),
    'MM-BrowseComp': ('https://github.com/MMBrowseComp/MM-BrowseComp.git', 'cf82a2390aa7a7ed3d96f72969d22548eff3e101'),
}


def verify(root):
    versions = {}
    for name, (_, expected) in REPOSITORIES.items():
        actual = subprocess.check_output(['git', '-C', str(Path(root) / name), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != expected:
            raise ValueError('Upstream revision differs: ' + name)
        versions[name] = actual
    return versions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/official_eval_dependencies'))
    a = p.parse_args()
    a.root.mkdir(parents=True, exist_ok=True)
    for name, (url, revision) in REPOSITORIES.items():
        path = a.root / name
        if not path.exists():
            subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', url, str(path)], check=True)
            subprocess.run(['git', '-C', str(path), 'checkout', '--detach', revision], check=True)
    print(verify(a.root))


if __name__ == '__main__':
    main()
