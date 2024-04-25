#!/bin/bash
set -euo pipefail

URL_PREFIX="https://echossupercoolnewtexttoimagemodel.com/test-images"

wget -O test-images/examples-100.pq $URL_PREFIX/examples-100.pq

mkdir -p test-images/capped-examples
for i in {0..2} ; do
    wget -O test-images/capped-examples/test-caps-"$i".parquet $URL_PREFIX/capped-examples/test-caps-"$i".parquet
done