#!/bin/sh

set -x
set -e
cd $(dirname $0)/..
TOP=$(pwd)

TEST_RUST=0
TEST_MB=0
TEST_PY=0

while [ "X$1" != "X" ] ; do
    case "$1" in
        --rust)
            TEST_RUST=1
            ;;
        --mb)
            TEST_MB=1
            ;;
        --py)
            TEST_PY=1
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
    shift
done

if [ "$TEST_RUST$TEST_MB$TEST_PY" = 000 ] ; then
    TEST_RUST=1
    TEST_MB=1
    TEST_PY=1
fi

# always check docs, it's quick
cd "$TOP"
./scripts/checklinks.sh

if [ "$TEST_RUST" = 1 ] ; then
    cd "$TOP"
    cargo fmt --check
    ./scripts/cbindgen.sh --check

    cargo clippy --workspace --all-targets --all-features -- -D warnings

    cargo build --locked

    (cd parser && cargo check --no-default-features)

    cargo test
    cargo test --release

    echo "Running sample_parser"
    (cd sample_parser && ./run.sh >/dev/null)
    (cd sample_parser && ./lark.sh test)

    (cd c_sample && make)
fi

if [ "$TEST_MB" = 1 ] ; then
    cd "$TOP"
    if [ -d ../jsonschemabench/maskbench/data ] ; then
        echo "MaskBench side by side"
        MB_PATH=../jsonschemabench/maskbench
    else
        mkdir -p tmp
        cd tmp
        if test -d jsonschemabench/maskbench/data ; then
            echo "MaskBench clone OK"
        else
            git clone -b main https://github.com/guidance-ai/jsonschemabench
        fi
        MB_PATH=tmp/jsonschemabench/maskbench
    fi

    cd "$TOP"
    if [ -d $MB_PATH/data ] ; then
        :
    else
        echo "MaskBench data missing"
        exit 1
    fi

    MB_PATH=$(realpath $MB_PATH)
    cd json_stats
    mkdir -p tmp
    cargo run --release -- \
        --llg-masks \
        --expected expected_maskbench.json \
        $MB_PATH/data
fi

if [ "$TEST_PY" = 1 ] ; then

cd "$TOP"

pip uninstall -y llguidance || :

if test -z "$CONDA_PREFIX" -a -z "$VIRTUAL_ENV" ; then
    if [ "X$CI" = "Xtrue" -o -f /.dockerenv ]; then
        echo "Building in CI with pip"
        pip install -v -e .
    else
        echo "No conda and no CI"
        exit 1
    fi
else
    maturin develop --release
fi

echo "Running mypy..."
(cd python && mypy)

PYTEST_FLAGS=

if test -f ../guidance/tests/unit/test_ll.py ; then
    echo "Guidance side by side"
    cd ../guidance
else
    mkdir -p tmp
    cd tmp
    if [ "X$CI" = "Xtrue" ] ; then
      PYTEST_FLAGS=-v
    fi
    if test -f guidance/tests/unit/test_ll.py ; then
        echo "Guidance clone OK"
    else
        echo "Cloning guidance"
        git clone -b llg_warnings https://github.com/guidance-ai/guidance
    fi
    cd guidance
    echo "Branch: $(git branch --show-current), Remote URL: $(git remote get-url origin), HEAD: $(git rev-parse HEAD)"
fi

# not so relevant anymore, we do it anyways below
# python -m pytest $PYTEST_FLAGS tests/unit/test_ll.py # main test

(cd "$TOP" && python -m pytest $PYTEST_FLAGS python/torch_tests/)
python -m pytest $PYTEST_FLAGS tests/unit/test_[lgmp]*.py tests/unit/library "$@"


fi # TEST_PY
