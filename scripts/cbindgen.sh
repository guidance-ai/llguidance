#!/bin/sh

CHECK=0
if [ "$1" = "--check" ]; then
    CHECK=1
    shift
fi

if cbindgen --version ; then
    echo "cbindgen is already installed"
else
    echo "Installing cbindgen"
    cargo install cbindgen
fi

cd "$(dirname "$0")/../parser"

mkdir -p tmp
cbindgen --config cbindgen.toml \
         --crate llguidance \
         --output tmp/llguidance0.h  > tmp/cbindgen.txt 2>&1

if [ $? -ne 0 ]; then
    echo "Failed to generate llguidance.h"
    cat tmp/cbindgen.txt
    exit 1
else
    # print warnings and errors, but skip "Skip" messages
    grep -v "Skip .*(not " tmp/cbindgen.txt

    cat tmp/llguidance0.h | \
        grep -v "\* # Safety" | \
        grep -v "\* This function should only be called from C code" \
    > tmp/llguidance.h

    if diff -u llguidance.h tmp/llguidance.h; then
        echo "llguidance.h is up to date"
    else
        if [ $CHECK -eq 1 ]; then
            echo "llguidance.h is out of date"
            exit 1
        else
            cp tmp/llguidance.h llguidance.h
            echo "Updated llguidance.h"
        fi
    fi
fi
