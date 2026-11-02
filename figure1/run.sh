#!/bin/bash
set -e
cd "$(dirname "$0")"
: "${DATA:?}" "${WORK:?}"
export DATA WORK
git submodule update --init --recursive .
export E=$PWD
mkdir -p $DATA $WORK
cd $DATA
bash $E/download.sh sift sift_q deep deep_q 'msturing[10M]'
C=https://comp21storage.z5.web.core.windows.net/comp21
Y=https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I
curl -sS --fail -r 8-1000000007 -o spacev.10M.100.i8.bin $C/spacev1b/spacev1b_base.i8bin
curl -sS --fail -r 8- -o spacev.29K.100.i8.query $C/spacev1b/query.i8bin
curl -sS --fail -r 8- -o msturing.100K.100.f32.query $C/MSFT-TURING-ANNS/query100K.fbin
curl -sS --fail -r 8- -o t2i.100K.200.f32.query $Y/query.public.100K.fbin
curl -sS --fail -r 8-8000000007 -o t2i.10M.200.f32.bin $Y/base.1B.fbin
cd -

make -C $E/parlayann/algorithms/vamana
cargo build --release --manifest-path $E/Cargo.toml --target-dir $E/target
mkdir -p $E/build
g++ -O3 -march=native -mcx16 -pthread -std=c++17 -DNDEBUG \
    -I $E/parlayann/algorithms -I $E/parlayann/parlaylib/include \
    $E/0012/sweep.cc -o $E/build/sweep
for ds in sift deep spacev msturing t2i; do
    bash $E/0012/run.sh crop $ds
    bash $E/0012/run.sh gt $ds
    bash $E/0012/run.sh build $ds
    bash $E/0012/run.sh filter $ds
    bash $E/0012/run.sh search $ds orig
    bash $E/0012/run.sh search $ds bidir
done
$E/target/release/recallqps $WORK/*.orig.log $WORK/*.bidir.log > $WORK/recallqps.csv

python3 $E/motivation_tables.py figure1 $WORK/recallqps.csv > recall_qps.csv
python3 $E/motivation_tables.py edges $WORK > edges_removed.csv
uv run plot.py
