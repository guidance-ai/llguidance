#!/usr/bin/env python3

import os
import re
import sys

exclude = {"tmp/", "node_modules/", "target/"}

# Read .gitmodules and extract paths to exclude
if os.path.exists(".gitmodules"):
    with open(".gitmodules", "r", encoding="utf-8") as f:
        for match in re.finditer(r"^\s*path = (.*)$", f.read(), re.MULTILINE):
            exclude.add(match.group(1) + "/")

links = {}
files = {}

# Read files from arguments
for file in sys.argv[1:]:
    if any(file.startswith(ex) for ex in exclude):
        continue

    abs_file = os.path.abspath(file)
    with open(abs_file, "r", encoding="utf-8") as f:
        content = f.read()
        files[abs_file] = content

    # Mask text inside triple backticks to preserve line numbers
    masked_content = re.sub(
        r"```.*?```", lambda m: "\n".join(["" for _ in m.group(0).split("\n")]), content, flags=re.DOTALL
    )

    for match in re.finditer(r"^#+ (.*)$", masked_content, re.MULTILINE):
        title = match.group(1)
        anchor = "#" + re.sub(r"[^a-z0-9 \-]+", "", title.lower()).replace(" ", "-")
        links[abs_file + anchor] = True

numerr = 0
numlinks = 0
numanchors = 0
numhttp = 0

for filename, content in files.items():
    line_no = 0

    # Mask text inside triple backticks to preserve line numbers
    masked_content = re.sub(
        r"```.*?```", lambda m: "\n".join(["" for _ in m.group(0).split("\n")]), content, flags=re.DOTALL
    )

    for line in masked_content.split("\n"):
        line_no += 1

        for match in re.finditer(r"\[([^\]]+)\]\(([^\)]+)\)", line):
            title, link = match.groups()

            if link.startswith(("https://", "http://", "mailto:")):
                numhttp += 1
                continue

            numlinks += 1
            if link.startswith("#"):
                link = filename + link

            linkfile, *anchor_parts = link.split("#", 1)
            linkfile = os.path.abspath(os.path.join(os.path.dirname(filename), linkfile))
            anchor = "#" + anchor_parts[0] if anchor_parts and not anchor_parts[0].startswith("L") else ""
            line_ref = anchor_parts[0] if anchor_parts and anchor_parts[0].startswith("L") else ""

            if not os.path.exists(linkfile):
                numerr += 1
                print(f"{filename}:{line_no}: Broken link '{title}': {link}")
                continue

            if anchor and linkfile + anchor not in links:
                numerr += 1
                print(f"{filename}:{line_no}: Broken link to anchor '{title}': {link}")
            elif line_ref:
                try:
                    line_number = int(line_ref[1:])
                    with open(linkfile, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        if line_number > len(lines) or line_number <= 0:
                            numerr += 1
                            print(f"{filename}:{line_no}: Broken link to line '{title}': {link}")
                except ValueError:
                    numerr += 1
                    print(f"{filename}:{line_no}: Invalid line reference '{title}': {link}")

if numerr > 0:
    print(f"Found {numerr} broken links")
    sys.exit(1)
else:
    print(f"Exclude: {', '.join(exclude)}")
    print(f"Checked {numlinks} links (incl. {numanchors} anchors). Skipped {numhttp} http links.")
