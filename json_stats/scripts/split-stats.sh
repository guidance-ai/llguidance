#!/bin/sh

splits="Github_easy
Github_hard
Github_medium
Github_trivial
Github_ultra
Glaiveai2K
Handwritten
JsonSchemaStore
Kubernetes
MCPspec
Snowplow
Synthesized
WashingtonPost
TOTAL"

set -e
mkdir -p tmp/splits
for split in $splits; do
    F="--filter $split--"
    if [ "$split" = "TOTAL" ]; then
        F=""
    fi
    for v in valid invalid ; do
        ./jstats.sh $JSB_DATA/unique_tests/ $F --only-$v
        mv tmp/test_total.json tmp/splits/$split-$v.json
    done

    ./jstats.sh $JSB_DATA/unique_tests/ $F --only-valid --compact
    mv tmp/test_total.json tmp/splits/$split-valid-compact.json
done
