#!/usr/bin/env python3

import os
import re
import subprocess


def get_toktrie_version():
    """Extracts the version of toktrie from its Cargo.toml using regex."""
    cargo_toml_path = os.path.join("toktrie", "Cargo.toml")
    with open(cargo_toml_path, "r") as f:
        content = f.read()

    match = re.search(r'^version\s*=\s*"(.*?)"', content, re.MULTILINE)
    if not match:
        raise ValueError("Could not find toktrie version in Cargo.toml")

    return match.group(1)


def update_dependency(crate, version):
    """Replaces workspace refs in [dependencies] with a version for publishing.

    Only rewrites ``{ workspace = true }`` lines within the [dependencies]
    section.  Lines in [dev-dependencies] (and other sections) are left
    untouched so that ``cargo publish`` correctly strips path-only
    dev-dependencies.

    We use simple line-by-line section tracking rather than a TOML library
    because tomllib (stdlib) is read-only, and we want to avoid adding a
    third-party dependency just for this script.
    """
    cargo_toml_path = os.path.join(crate, "Cargo.toml")
    with open(cargo_toml_path, "r") as f:
        lines = f.readlines()

    in_dependencies = False
    workspace_re = re.compile(r"^(\S+)\s*=\s*\{ workspace = true \}")
    updated_lines = []
    for line in lines:
        if line.strip().startswith("["):
            in_dependencies = line.strip() == "[dependencies]"
        if in_dependencies:
            m = workspace_re.match(line)
            if m:
                dep_name = m.group(1)
                line = f'{dep_name} = {{ version = "{version}" }}\n'
        updated_lines.append(line)

    with open(cargo_toml_path, "w") as f:
        f.writelines(updated_lines)

    return "".join(lines)  # Return original content for restoration


def restore_dependency(crate, original_content):
    """Restores the original Cargo.toml content."""
    cargo_toml_path = os.path.join(crate, "Cargo.toml")
    with open(cargo_toml_path, "w") as f:
        f.write(original_content)


def publish_crate(crate):
    """Runs `cargo publish` in the specified crate directory."""
    subprocess.run(["cargo", "publish", "--allow-dirty"], cwd=crate, check=True)


def main():
    toktrie_version = get_toktrie_version()

    # Publish toktrie first
    print(f"Publishing toktrie v{toktrie_version}...")
    publish_crate("toktrie")

    # Publish dependent crates
    for crate in ["toktrie_hf_tokenizers", "toktrie_hf_downloader", "toktrie_tiktoken", "parser"]:
        print(f"Updating {crate} to use toktrie v{toktrie_version}...")
        original_content = update_dependency(crate, toktrie_version)

        try:
            print(f"Publishing {crate}...")
            publish_crate(crate)
        finally:
            print(f"Restoring original {crate} Cargo.toml...")
            restore_dependency(crate, original_content)

    print("All crates published successfully.")


if __name__ == "__main__":
    main()
