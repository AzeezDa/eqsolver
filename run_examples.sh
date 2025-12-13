#!/bin/sh

cd examples
for file in *; do
    if [ -f "$file" ]; then
        cargo run --example "${file%.*}"
    fi
done
