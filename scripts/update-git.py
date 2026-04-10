#!/usr/bin/env python3

import re
import subprocess
import sys


def get_version_from_cargo_toml(cargo_toml_path):
    try:
        with open(cargo_toml_path, "r") as file:
            cargo_toml_contents = file.read()

        # Match version field in Cargo.toml
        match = re.search(r'\bversion\s*=\s*"([^"]+)"', cargo_toml_contents)

        if match:
            return match.group(1)
        else:
            print(f"Error: Version not found in {cargo_toml_path}.")
            return None
    except FileNotFoundError:
        print(f"Error: {cargo_toml_path} not found.")
        return None
    except Exception as e:
        print(f"Error reading {cargo_toml_path}: {e}")
        return None


def update_cargo_toml(cargo_toml_path, derivre_version):
    try:
        with open(cargo_toml_path, "r") as file:
            cargo_toml_contents = file.read()

        # Patterns for replacing the version in Cargo.toml
        derivre_pattern = r'(derivre\s*=\s*\{[^}]*version\s*=\s*")[^"]*(")'

        cargo_toml_contents = re.sub(
            derivre_pattern,
            lambda m: m.group(1) + derivre_version + m.group(2),
            cargo_toml_contents,
        )

        # Write the updated contents back to the Cargo.toml file
        with open(cargo_toml_path, "w") as file:
            file.write(cargo_toml_contents)

        print(f"{cargo_toml_path} updated successfully.")
    except FileNotFoundError:
        print(f"Error: {cargo_toml_path} not found.")
    except Exception as e:
        print(f"Error updating {cargo_toml_path}: {e}")


# Get the version from ../derivre/Cargo.toml
derivre_version = get_version_from_cargo_toml("../derivre/Cargo.toml")

# Check if the version was retrieved successfully
if not derivre_version:
    print("Error retrieving version. Exiting.")
    sys.exit(1)

# List of Cargo.toml paths to update
cargo_toml_paths = [
    "parser/Cargo.toml",
]

# Update each Cargo.toml file
for cargo_toml_path in cargo_toml_paths:
    update_cargo_toml(cargo_toml_path, "=" + derivre_version)

subprocess.run(["cargo", "check"], check=True)

print("All Cargo.toml files updated and cargo fetch run successfully.")
