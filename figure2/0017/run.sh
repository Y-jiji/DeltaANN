#!/bin/bash
# run.sh <compile|cut|build|hop> [ds] [degree]: candidate fates under SaturationPrune
# compile: crop, ParlayANN's neighbors and twohop into $E/target/release, the vamana dir and $E/build
# cut: the dataset's 20M rows 0 to 10M as the base and 10M to 10M + 10K as the points, headered, in $W
# build: ParlayANN R96 L128 alpha 1.2 one pass over the base into $I/parlay_<ds>_10m_d96.graph
# hop: twohop over the base, the points and the R96 graph trimmed to <degree> into $L/twohop017_<ds>_d<degree>.csv
set -euo pipefail

S=${DATA:?set DATA to the dataset directory}
I=${WORK:?set WORK to a scratch directory}/index
L=$WORK/logs
W=$WORK
E=${E:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

declare -A RAW=([sift]=sift.20M.128.bin [deep]=deep.20M.96.bin [spacev]=spacev.20M.100.shuf.bin
                [msturing]=msturing.20M.100.f32.bin [t2i]=t2i.20M.200.bin)
declare -A DIM=([sift]=128 [deep]=96 [spacev]=100 [msturing]=100 [t2i]=200)
declare -A SIZE=([sift]=1 [deep]=4 [spacev]=1 [msturing]=4 [t2i]=4)
declare -A TYPE=([sift]=uint8 [deep]=float [spacev]=int8 [msturing]=float [t2i]=float)

compile() {
    cargo build --release --manifest-path $E/Cargo.toml --target-dir $E/target --bin crop
    make -C $E/parlayann/algorithms/vamana
    mkdir -p $E/build
    g++ -DHOMEGROWN -pthread -mcx16 -O3 -std=c++17 -march=native -DNDEBUG \
        -I $E/parlayann/algorithms -I $E/parlayann/parlaylib/include \
        "$E/0017/twohop.cc" -o $E/build/twohop -ldl
}

cut() {
    $E/target/release/crop $S/${RAW[$1]} $W/$1.10M.bin 10000000 ${DIM[$1]} ${SIZE[$1]}
    $E/target/release/crop $S/${RAW[$1]} $W/$1.points.bin 10000 ${DIM[$1]} ${SIZE[$1]} 10000000
}

build() {
    mkdir -p $I $L
    $E/parlayann/algorithms/vamana/neighbors -R 96 -L 128 -alpha 1.2 -two_pass 0 \
        -data_type ${TYPE[$1]} -dist_func Euclidian -base_path $W/$1.10M.bin \
        -graph_outfile $I/parlay_$1_10m_d96.graph > $L/parlay017_$1_d96.log 2>&1
}

hop() {
    mkdir -p $L
    $E/build/twohop ${TYPE[$1]} $W/$1.10M.bin $W/$1.points.bin $I/parlay_$1_10m_d96.graph $2 \
        > $L/twohop017_$1_d$2.csv 2> $L/twohop017_$1_d$2.log
}

"$@"
