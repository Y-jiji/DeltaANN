#!/bin/bash
# gcpprep.sh <commit> <name> [spacev|msturing]: #027's reordered stream on a gcloud Batch n2-highcpu-80 VM: the
# bucket's cluster-ordered <ds>k/<ds>.201M.k128.raw (spacev by default) with its 100M base kept as is and its 101M
# inserted rows put in a random order by 0027/shuffle.py (seed 27), uploaded to <ds>s/ with the source-row index, then
# gtbrute's ground truth of the queries over the 201M reordered rows, one block per 1M rows, to <ds>s/; logs to
# runs/<name>
set -euo pipefail
export HOME=/root
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
B=${BUCKET:?set BUCKET to a GCS bucket holding the inputs and the sources}
E=/var/tmp/eval
W=/var/tmp/work
FNCT=bf5d938
SEED=27
DS=${3:-spacev}
case $DS in
    spacev) KIND=i8 WIDTH=100 ;;
    msturing) KIND=f32 WIDTH=400 ;;
    *) echo "dataset $DS, want spacev|msturing"; exit 1 ;;
esac
log() { echo "PHASE $* $(date +%T)"; }
mkdir -p $W $E/fnct
# [1] prepare: the inputs in the background, the toolchain, the sources, gtbrute
gcloud storage cp $B/${DS}k/$DS.201M.k128.raw $B/${DS}k/$DS.query $W/ &
C=$!
apt-get -o DPkg::Lock::Timeout=600 -qq update
DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -y -qq --no-install-recommends install \
    build-essential curl ca-certificates time
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
curl -LsSf https://astral.sh/uv/install.sh | sh
tar xf /var/tmp/boot/eval-$1.tar -C $E
gcloud storage cat $B/src/fnct-$FNCT.tar | tar x -C $E/fnct
cd $E
cargo build --release --bin gtbrute
wait $C
log built
# [2] the reordered stream: rows [0, 100M) as they are, rows [100M, 201M) shuffled
/usr/bin/time -v uv run --script 0027/shuffle.py $W/$DS.201M.k128.raw $W/$DS.201M.s.raw $W/$DS.201M.s.idx.npy \
    $WIDTH 100000000 $SEED > $W/shuffle.log 2>&1
cmp -n $((100000000 * WIDTH)) $W/$DS.201M.k128.raw $W/$DS.201M.s.raw
rm -f $W/$DS.201M.k128.raw
gcloud storage cp $W/$DS.201M.s.raw $W/$DS.201M.s.idx.npy $B/${DS}s/
gcloud storage cp -q $W/shuffle.log $B/runs/$2/
log shuffled
# [3] the ground truth of the queries over the 201M reordered rows
/usr/bin/time -v $E/target/release/gtbrute $KIND 100 $W/$DS.201M.s.raw $W/$DS.query $W/$DS.201M.s.gt.arrows $W/brute \
    > $W/gt.log 2>&1
gcloud storage cp $W/$DS.201M.s.gt.arrows $B/${DS}s/
gcloud storage cp -q $W/gt.log $B/runs/$2/
log done
