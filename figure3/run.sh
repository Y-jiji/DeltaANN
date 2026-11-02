#!/bin/bash
set -e
cd "$(dirname "$0")"
: "${DATA:?}" "${WORK:?}"
export DATA WORK
git submodule update --init --recursive .
export E=$PWD
mkdir -p $DATA $WORK
C=https://comp21storage.z5.web.core.windows.net/comp21
curl -sS --fail -r 8- -o $DATA/spacev.29K.100.i8.query $C/spacev1b/query.i8bin
curl -sS --fail -r 8- -o $DATA/msturing.100K.100.f32.query $C/MSFT-TURING-ANNS/query100K.fbin

make -C $E/parlayann/algorithms/vamana
cargo build --release --manifest-path $E/Cargo.toml --target-dir $E/target
bash $E/0019/run.sh build
R="bash $E/0019/run.sh"
for ds in spacev msturing; do
    $R download $ds
    $R reorder $ds
    $R slice $ds
    $R codebook $ds A
    $R codebook $ds B
    $R truth $ds
    $R graph $ds AB
    $R graph $ds BB
    $R graph $ds AE
    $R search $ds AB pq;   $R wide $ds AB
    $R search $ds AB pq A; $R wide $ds AB A
    $R search $ds AE pq;   $R wide $ds AE
    $R search $ds BB pq;   $R wide $ds BB
    K=$WORK/k128
    python3 $E/0019/recallqps.py \
        $K/$ds.search.AB.pq.log $K/$ds.search.AB.pq.wide.log \
        $K/$ds.search.AB.pq.A.log $K/$ds.search.AB.pq.A.wide.log \
        $K/$ds.search.AE.pq.log $K/$ds.search.AE.pq.wide.log \
        $K/$ds.search.BB.pq.log $K/$ds.search.BB.pq.wide.log \
        > $WORK/recallqps019_$ds.csv
done

python3 $E/motivation_tables.py figure3 $WORK > relative_qps.csv
uv run plot.py
