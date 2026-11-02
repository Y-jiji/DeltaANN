#!/bin/bash
# gcpprep.sh <commit> <name> <deep|spacevk|msturingk> <graph|gt|both>: a 100M base + 100M insert workload's inputs on
# a gcloud Batch n2-highmem-64 VM, from the bucket's <ds>/ slices: the ParlayANN Vamana graph of the 100M base (R 64,
# L 128, alpha 1.2, two passes, as sift.100M's), flattened by 0002/flatgraph.py and made bidirectional by bidir,
# and/or gtbrute's ground truth of the queries over the 201M rows; the outputs to the bucket's <ds>/, logs to
# runs/<name>
set -euo pipefail
export HOME=/root
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
B=${BUCKET:?set BUCKET to a GCS bucket holding the inputs and the sources}
E=/var/tmp/eval
W=/var/tmp/work
FNCT=bf5d938
PARLAY=a48c6a3
PLIB=c655b8f
DS=$3
STEPS=$4
case $DS in
    deep) KIND=f32 TYPE=float RAW=deep.201M.96.f32.bin BASE=deep.100M.96.fbin QUERY=deep.10K.96.f32.query DIM=96
        OUT=deep.100M GT=deep.gt.arrows ;;
    spacevk) KIND=i8 TYPE=int8 RAW=spacev.201M.k128.raw BASE=spacev.100M.k128.bin QUERY=spacev.query DIM=100
        OUT=spacev.100M.k128 GT=spacev.201M.k128.gt.arrows ;;
    msturingk) KIND=f32 TYPE=float RAW=msturing.201M.k128.raw BASE=msturing.100M.k128.bin QUERY=msturing.query DIM=100
        OUT=msturing.100M.k128 GT=msturing.201M.k128.gt.arrows ;;
    *) echo "dataset $DS, want deep|spacevk|msturingk"; exit 1 ;;
esac
log() { echo "PHASE $* $(date +%T)"; }
mkdir -p $W $E/fnct $E/parlayann
# [1] prepare: the toolchain, the sources, the tools
apt-get -o DPkg::Lock::Timeout=600 -qq update
DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -y -qq --no-install-recommends install \
    build-essential curl ca-certificates python3 time
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
curl -LsSf https://astral.sh/uv/install.sh | sh
tar xf /var/tmp/boot/eval-$1.tar -C $E
gcloud storage cat $B/src/fnct-$FNCT.tar | tar x -C $E/fnct
gcloud storage cat $B/src/parlayann-$PARLAY.tar | tar x -C $E/parlayann
mkdir -p $E/parlayann/parlaylib
gcloud storage cat $B/src/parlaylib-$PLIB.tar | tar x -C $E/parlayann/parlaylib
cd $E
cargo build --release --bin gtbrute --bin bidir
make -C $E/parlayann/algorithms/vamana neighbors
log built
# [2] the graph: Vamana over the 100M base, flattened, mutual edges only, as sift.100M.flat.bidir
if [ $STEPS != gt ]; then
    gcloud storage cp $B/$DS/$BASE $W/
    /usr/bin/time -v $E/parlayann/algorithms/vamana/neighbors -R 64 -L 128 -alpha 1.2 -num_passes 2 -data_type $TYPE \
        -dist_func Euclidian -base_path $W/$BASE -graph_outfile $W/$OUT.graph.orig > $W/vamana.log 2>&1
    gcloud storage cp -q $W/vamana.log $B/runs/$2/
    log graph
    uv run "0002/flatgraph.py" $W/$OUT.graph.orig $W/$OUT.flat.orig
    rm -f $W/$OUT.graph.orig $W/$BASE
    $E/target/release/bidir flat 64 $W/$OUT.flat.orig $W/$OUT.flat.bidir
    gcloud storage cp $W/$OUT.flat.bidir $B/$DS/
    rm -f $W/$OUT.flat.orig
    log bidir
fi
# [3] the ground truth of the queries over the 201M rows, one block per 1M rows
if [ $STEPS != graph ]; then
    gcloud storage cp $B/$DS/$RAW $B/$DS/$QUERY $W/
    /usr/bin/time -v $E/target/release/gtbrute $KIND $DIM $W/$RAW $W/$QUERY $W/$GT $W/brute > $W/gt.log 2>&1
    gcloud storage cp $W/$GT $B/$DS/
    gcloud storage cp -q $W/gt.log $B/runs/$2/
    log gt
fi
log done
