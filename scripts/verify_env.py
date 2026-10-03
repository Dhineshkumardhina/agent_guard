#!/usr/bin/env python
"""verify_env.py — AgentGuard environment verification script.

Checks that all required Python packages are importable and prints
a full system provenance report.  Run before starting any experiment.

Usage:
    python scripts/verify_env.py
"""

import sys
import importlib

REQUIRED_PACKAGES = [
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
    ("pydantic", "pydantic"),
    ("pydantic_settings", "pydantic-settings"),
    ("sqlalchemy", "sqlalchemy"),
    ("networkx", "networkx"),
    ("numpy", "numpy"),
    ("dotenv", "python-dotenv"),
    ("yaml", "pyyaml"),
]

OPTIONAL_PACKAGES = [
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("torch", "torch"),
    ("torch_geometric", "torch-geometric"),
]


def check_import(module: str, pkg_name: str, required: bool = True) -> bool:
    """Attempt to import module and report status."""
    try:
        mod = importlib.import_module(module)
        version = getattr(mod, "__version__", "?")
        status = "✓" if required else "○"
        print(f"  {status}  {pkg_name:<28} {version}")
        return True
    except ImportError:
        status = "✗" if required else "–"
        note = "(required)" if required else "(optional, not installed)"
        print(f"  {status}  {pkg_name:<28} NOT FOUND {note}")
        return False


def main() -> int:
    print("=" * 56)
    print("  AgentGuard — Environment Verification")
    print("=" * 56)
    print(f"\nPython: {sys.version}")
    print(f"Interpreter: {sys.executable}\n")

    print("Required packages:")
    required_ok = all(
        check_import(mod, pkg, required=True) for mod, pkg in REQUIRED_PACKAGES
    )

    print("\nOptional packages (Phase 8+):")
    for mod, pkg in OPTIONAL_PACKAGES:
        check_import(mod, pkg, required=False)

    print()
    if required_ok:
        print("✓ All required packages are available.")
        print("✓ Environment is ready for Phase 1.")
    else:
        print("✗ Some required packages are missing.")
        print("  Run: pip install -r requirements.txt")
        return 1

    # Run provenance capture
    try:
        from ml.utils.reproducibility import get_system_provenance
        prov = get_system_provenance()
        print("\nSystem provenance:")
        for k, v in prov.items():
            print(f"  {k}: {v}")
    except Exception as exc:
        print(f"\n[WARNING] Could not capture provenance: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
