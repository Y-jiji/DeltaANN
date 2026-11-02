#!/bin/bash
# run.sh <crop|gt|build|filter|search> <ds> [orig|bidir]
set -euo pipefail

D=${WORK:?set WORK to a scratch directory}
S=${DATA:?set DATA to the dataset directory}
E=${E:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}

declare -A BASE=([sift]=sift.1B.128.u8.bin [deep]=deep.1B.96.f32.bin [spacev]=spacev.10M.100.i8.bin
                 [msturing]=msturing.10M.100.f32.bin [t2i]=t2i.10M.200.f32.bin)
declare -A QUERY=([sift]=sift.10K.128.u8.query [deep]=deep.10K.96.f32.query [spacev]=spacev.29K.100.i8.query
                  [msturing]=msturing.100K.100.f32.query [t2i]=t2i.100K.200.f32.query)
declare -A NQ=([sift]=10000 [deep]=10000 [spacev]=29316 [msturing]=100000 [t2i]=100000)
declare -A DIM=([sift]=128 [deep]=96 [spacev]=100 [msturing]=100 [t2i]=200)
declare -A SIZE=([sift]=1 [deep]=4 [spacev]=1 [msturing]=4 [t2i]=4)
declare -A KIND=([sift]=u8 [deep]=f32 [spacev]=i8 [msturing]=f32 [t2i]=f32)
declare -A TYPE=([sift]=uint8 [deep]=float [spacev]=int8 [msturing]=float [t2i]=float)

crop() {
    $E/target/release/crop $S/${BASE[$1]} $D/$1.10M.bin 10000000 ${DIM[$1]} ${SIZE[$1]}
    $E/target/release/crop $S/${QUERY[$1]} $D/$1.query ${NQ[$1]} ${DIM[$1]} ${SIZE[$1]}
}

gt() {
    $E/target/release/gtbrute ${KIND[$1]} $D/$1.10M.bin $D/$1.query $D/$1.10M.gt $D/$1.brute
    rm -f $D/$1.brute
}

build() {
    $E/parlayann/algorithms/vamana/neighbors -R 64 -L 128 -alpha 1.2 -num_passes 2 \
        -data_type ${TYPE[$1]} -dist_func Euclidian \
        -base_path $D/$1.10M.bin -graph_outfile $D/$1.graph.orig
}

filter() {
    $E/target/release/bidir $1 $D/$1.graph.orig $D/$1.graph.bidir
}

search() {
    $E/build/sweep $1.$2 ${TYPE[$1]} $D/$1.10M.bin $D/$1.query $D/$1.10M.gt $D/$1.graph.$2 > $D/$1.$2.log
}

"$@"
