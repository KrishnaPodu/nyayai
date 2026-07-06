"""
checks that the key packages actually resolved to the versions we pinned.
uv can silently pick a different version if the pin in pyproject.toml
is wrong or if a sub-dependency forces something else.
run with: uv run python test_deps.py
"""

import importlib.metadata as md


def check(package_name, expected_prefix):
    try:
        version = md.version(package_name)
        ok = version.startswith(expected_prefix)
        status = "OK" if ok else "MISMATCH"
        print(f"[{status}] {package_name}: {version} (expected prefix: {expected_prefix})")
    except md.PackageNotFoundError:
        print(f"[MISSING] {package_name} is not installed")


check("torch", "2.4.1")
check("transformers", "4.")  # anything 4.56.1+ but <6.0.0, just confirming it's not 6.x or ancient
check("surya-ocr", "0.9.3")

print()
print("also manually confirm:")
print("  - transformers version is >=4.56.1 (check above isn't ancient like 4.30)")
print("  - python version is 3.10 (run: python --version)")