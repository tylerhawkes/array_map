#!/usr/bin/env bash
set -ex

cd $(dirname ${BASH_SCRIPT[0]})

function publish() {
  cargo clippy ${1}
  cargo build ${1}
  cargo test ${1}
  cargo publish --registry crates-io
}

cd array_map_derive
publish
cd ..
publish --workspace