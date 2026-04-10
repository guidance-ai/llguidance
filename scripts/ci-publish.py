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


PUBLISHABLE_CRATES = {
    "toktrie", "toktrie_hf_tokenizers", "toktrie_hf_downloader",
    "toktrie_tiktoken", "llguidance",
}


def update_dependency(crate, version):
    """Replaces workspace refs for publishable deps in Cargo.toml.

    Only dependencies whose name is in PUBLISHABLE_CRATES are rewritten.
    Other workspace references (e.g. llg_test_utils, rand) are left
    untouched so that cargo publish correctly strips path-only
    dev-dependencies.
    """
    cargo_toml_path = os.path.join(crate, "Cargo.toml")
    with open(cargo_toml_path, "r") as f:
        content = f.read()

    def replace_if_publishable(m):
        dep_name = m.group(1)
        if dep_name in PUBLISHABLE_CRATES:
            return f'{dep_name} = {{ version = "{version}" }}'
        return m.group(0)

    updated_content = re.sub(
        r'^(\S+)\s*=\s*\{ workspace = true \}',
        replace_if_publishable,
        content,
        flags=re.MULTILINE,
    )

    with open(cargo_toml_path, "w") as f:
        f.write(updated_content)

    return content  # Return original content for restoration


def restore_dependency(crate, original_content):
    """Restores the original Cargo.toml content."""
    cargo_toml_path = os.path.join(crate, "Cargo.toml")
    with open(cargo_toml_path, "w") as f:
        f.write(original_content)


def publish_crate(crate):
    """Runs `cargo publish` in the specified crate directory."""
    subprocess.run(["cargo", "publish", "--allow-dirty"],
                   cwd=crate,
                   check=True)


def main():
    toktrie_version = get_toktrie_version()

    # Publish toktrie first
    print(f"Publishing toktrie v{toktrie_version}...")
    publish_crate("toktrie")

    # Publish dependent crates
    for crate in [
            "toktrie_hf_tokenizers", "toktrie_hf_downloader",
            "toktrie_tiktoken", "parser"
    ]:
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
