#!/usr/bin/env python
import ast
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

ROOT = Path(__file__).parent
THIS_FILE = Path(__file__).name

def find_imports_in_file(path: Path) -> set[str]:
    imports = set()
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return imports

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                imports.add(top)
        elif isinstance(node, ast.ImportFrom):
            # skip relative imports: from .foo import bar
            if node.level != 0:
                continue
            if node.module is None:
                continue
            top = node.module.split(".")[0]
            imports.add(top)
    return imports

def main():
    modules: set[str] = set()

    # scan all .py files in this directory (non-recursive)
    for py_file in ROOT.glob("*.py"):
        if py_file.name == THIS_FILE:
            continue
        modules |= find_imports_in_file(py_file)

    # resolve versions via installed packages
    requirements: dict[str, str] = {}
    for mod in sorted(modules):
        try:
            ver = version(mod)
        except PackageNotFoundError:
            # probably stdlib or not installed as a package – skip
            continue
        requirements[mod] = ver

    # write requirements.txt
    req_path = ROOT / "requirements.txt"
    with req_path.open("w", encoding="utf-8") as f:
        for name in sorted(requirements):
            f.write(f"{name}=={requirements[name]}\n")

    print(f"Written {req_path} with {len(requirements)} packages.")

if __name__ == "__main__":
    main()
