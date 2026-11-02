#!/bin/bash
# run.sh <download|reorder|slice|codebook|truth|build|graph|search> [spacev|msturing] [A|B|exact|AB|BB]
# slice: rows 0 to 10M (A) and 10M to 20M (B) of the reordered file, raw, B and the queries also with
# a ParlayANN header; codebook: pqtrain BAND 8 BITS 8 on A or B; truth: gtbrute over B, merged to
# ParlayANN ground truth; graph: exact (ParlayANN), AB or BB (sdcbuild with codebook A or B) over B;
# graph AE: sdcbuild with codebook A and the prune by exact distance
# search <ds> <graph> <pq|exact> [A|B]: pqsweep of a graph, traversal by a codebook's ADC (B by
# default) or exact distance
# wide <ds> <graph> [A|B]: pqsweep by ADC at beams 500, 1000, 2000
set -euo pipefail

S=${DATA:?set DATA to the dataset directory}
K=${WORK:?set WORK to a scratch directory}/k128
E=${E:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
ROWS=10000000

declare -A BASE=([spacev]=spacev.1B.100.i8.bin [msturing]=msturing.1B.100.f32.bin)
declare -A QUERY=([spacev]=spacev.29K.100.i8.query [msturing]=msturing.100K.100.f32.query)
declare -A NQ=([spacev]=29316 [msturing]=100000)
declare -A SIZE=([spacev]=1 [msturing]=4)
declare -A KIND=([spacev]=i8 [msturing]=f32)
declare -A TYPE=([spacev]=int8 [msturing]=float)

download() {
    cd $S
    bash $E/download.sh $1
}

reorder() {
    $E/target/release/reorder $1 $S/${BASE[$1]} $S/$1.1B.k128.bin $S/$1.1B.k128.perm \
        $S/$1.1B.k128.centroids > $S/$1.1B.k128.sizes.csv
}

slice() {
    local bytes=$((ROWS * 100 * ${SIZE[$1]}))
    mkdir -p $K
    dd if=$S/$1.1B.k128.bin of=$K/$1.A.raw bs=16M iflag=count_bytes count=$bytes
    dd if=$S/$1.1B.k128.bin of=$K/$1.B.raw bs=16M iflag=skip_bytes,count_bytes skip=$bytes count=$bytes
    $E/target/release/crop $K/$1.B.raw $K/$1.B.bin $ROWS 100 ${SIZE[$1]}
    $E/target/release/crop $S/${QUERY[$1]} $K/$1.query ${NQ[$1]} 100 ${SIZE[$1]}
}

codebook() {
    DATASET=$1 DATA=$K/$1.$2.raw BASE=$ROWS BAND=8 BITS=8 OUTPUT=$K/$1.$2.book $E/target/release/pqtrain
}

truth() {
    $E/target/release/gtbrute ${KIND[$1]} 100 $K/$1.B.raw $S/${QUERY[$1]} $K/$1.B.arrows $K/$1.scratch
    rm -f $K/$1.scratch
    $E/target/release/gtmerge $K/$1.B.arrows $K/$1.B.gt
}

build() {
    local flags="-std=c++20 -O3 -DNDEBUG -march=native -mcx16 -pthread"
    local incs="-I $E/parlayann/algorithms -I $E/parlayann/parlaylib/include -I $E/0019"
    mkdir -p $E/build
    g++ $flags $incs -o $E/build/sdcbuild "$E/0019/sdcbuild.cc"
    g++ $flags $incs -o $E/build/pqsweep "$E/0019/pqsweep.cc"
}

graph() {
    if [ $2 = exact ]; then
        $E/parlayann/algorithms/vamana/neighbors -R 32 -L 128 -alpha 1.2 -num_passes 2 \
            -data_type ${TYPE[$1]} -dist_func Euclidian -base_path $K/$1.B.bin -graph_outfile $K/$1.graph.exact
    else
        local prune=sdc
        [ ${2:1:1} = E ] && prune=exact
        $E/build/sdcbuild ${TYPE[$1]} $K/$1.B.bin $K/$1.${2:0:1}.book $K/$1.graph.$2 $prune
    fi
}

search() {
    local book=${4:-B} tag=""
    [ $book = B ] || tag=.$book
    $E/build/pqsweep $1.$2 ${TYPE[$1]} $3 $K/$1.B.bin $K/$1.query $K/$1.B.gt $K/$1.$book.book \
        $K/$1.graph.$2 > $K/$1.search.$2.$3$tag.log
}

wide() {
    local book=${3:-B} tag=""
    [ $book = B ] || tag=.$book
    BEAMS=500,1000,2000 $E/build/pqsweep $1.$2 ${TYPE[$1]} pq $K/$1.B.bin $K/$1.query $K/$1.B.gt \
        $K/$1.$book.book $K/$1.graph.$2 > $K/$1.search.$2.pq$tag.wide.log
}

"$@"
