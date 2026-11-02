#!/bin/bash
# gcpfull.sh <commit> <name> <ours|odin|conda> <sift|deep|spacevk|spacevs|msturingk|msturings> <grow|churn>: a full run
# on a gcloud Batch z3-highmem-8-highlssd VM over the 100M base, 100 batches of 1M inserts, each beside 1M L=50
# searches, churn also deleting the oldest 1M rows a batch, the queries swept every 10 batches; spacevk and msturingk
# the 128-cluster orders, spacevs and msturings their bases with the inserted rows shuffled (#027) on spacevk's and
# msturingk's graph and queries, their points awaited until they land; ours: instantmix (fnct bf5d938) with watersearch
# at width 2, sift opening the bucket's snapshot in place of the load, the others loading and writing their codebook to
# the bucket's <ds>/ under grow, to runs/<name> under churn; odin and conda: odinmix and condamix (pipeann 5147dab,
# upstream's one background writer), sift on our pivots, the others on pivots converted from the grow run's codebook
# once it lands, fnct's beam search for queries and odin's inserts at width 2 and 64 threads, odin merging deletes
# every 1.5M under churn, the index built first, then 512 GiB of zeros appended to its disk index; a MON line every 30
# s (epoch, busy and idle jiffies, SSD reads and writes completed, used MB); the log, MON and arrows go to runs/<name>
# every 10 minutes and at the end
set -euo pipefail
export HOME=/root
export PATH=$HOME/.cargo/bin:$PATH
B=${BUCKET:?set BUCKET to a GCS bucket holding the inputs and the sources}
E=/var/tmp/eval
S=/mnt/lssd
PIPEANN=5147dab
FNCT=bf5d938
FRESH=2c185ab
ARM=$3
DATA=$4
MODE=$5
O=$B/runs/$2
D=$(readlink -f /dev/disk/by-id/google-local-nvme-ssd-0)
SRC=$DATA
M=glibc.malloc.mmap_threshold=33554432:glibc.malloc.trim_threshold=17179869184:glibc.malloc.top_pad=268435456
case $DATA in
    sift) LABEL=sift DIM=128 BAND=8 POINTS=sift.201M.128.u8.bin GRAPH=sift.100M.flat.bidir QUERY=sift.10K.128.u8.query ;;
    deep) LABEL=deep DIM=96 BAND=4 POINTS=deep.201M.96.f32.bin GRAPH=deep.100M.flat.bidir QUERY=deep.10K.96.f32.query ;;
    spacevk) LABEL=spacev DIM=100 BAND=8 POINTS=spacev.201M.k128.raw GRAPH=spacev.100M.k128.flat.bidir QUERY=spacev.query ;;
    spacevs) LABEL=spacev DIM=100 BAND=8 POINTS=spacev.201M.s.raw GRAPH=spacev.100M.k128.flat.bidir QUERY=spacev.query
        SRC=spacevk ;;
    msturingk) LABEL=msturing DIM=100 BAND=8 POINTS=msturing.201M.k128.raw GRAPH=msturing.100M.k128.flat.bidir QUERY=msturing.query ;;
    msturings) LABEL=msturing DIM=100 BAND=8 POINTS=msturing.201M.s.raw GRAPH=msturing.100M.k128.flat.bidir QUERY=msturing.query
        SRC=msturingk ;;
    *) echo "dataset $DATA, want sift|deep|spacevk|spacevs|msturingk|msturings"; exit 1 ;;
esac
case $ARM$MODE in
    oursgrow|odingrow) TAG=full ;;
    ourschurn|odinchurn|condachurn) TAG=churn ;;
    *) echo "arm $ARM under $MODE, want ours|odin under grow, ours|odin|conda under churn"; exit 1 ;;
esac
BOOK=$LABEL.100M.instwater.book
PIVOTS=$LABEL.100M.instwater.pq_pivots.bin
T=$ARM${TAG}_$DATA
# [1] prepare: the local SSD, the inputs in the background, the toolchain and sources as gcpbuild.sh
mkfs.ext4 -q -F $D
mkdir -p $S
mount $D $S
systemctl stop unattended-upgrades apt-daily.timer apt-daily-upgrade.timer || true
sysctl -w fs.aio-max-nr=1048576
sysctl -w kernel.core_pattern=$S/core.%e.%p
ulimit -n 1048576
ulimit -c unlimited
(until gcloud storage ls $B/$DATA/$POINTS > /dev/null 2>&1; do sleep 60; done
    gcloud storage cp $B/$DATA/$POINTS $B/$SRC/$GRAPH $B/$SRC/$QUERY $S/) &
C=$!
apt-get -o DPkg::Lock::Timeout=600 -qq update
DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -y -qq --no-install-recommends install \
    build-essential cmake libboost-dev libaio-dev liburing-dev libopenblas-dev libeigen3-dev curl ca-certificates
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
mkdir -p /usr/local/include/mkl $E/pipeann $E/fnct $E/freshdiskann
touch /usr/local/include/mkl/mkl_cblas.h
ln -sf "$(find /usr/lib -name libopenblas.so | head -1)" /usr/local/lib/libcblas.so
export CPATH=/usr/include/eigen3:$(dirname "$(find /usr/include -name cblas.h | head -1)")
tar xf /var/tmp/boot/eval-$1.tar -C $E
gcloud storage cat $B/src/pipeann-$PIPEANN.tar | tar x -C $E/pipeann
gcloud storage cat $B/src/fnct-$FNCT.tar | tar x -C $E/fnct
gcloud storage cat $B/src/fresh-$FRESH.tar | tar x -C $E/freshdiskann
# [2] build: the arm's driver only
cd $E
if [ $ARM != ours ]; then
    (cd $E/pipeann/third_party/liburing && ./configure && make -C src -j 8)
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DIO_ENGINE=aio -DUSE_TCMALLOC=OFF
    cmake --build build --target ${ARM}mix -j 8
else
    cargo build --release --bin instantmix
fi
wait $C
echo "PHASE built $(date +%T)"
# [3] monitor and periodic upload
(while true; do
    echo "MON $(date +%s) $(awk '/^cpu / { print $2 + $3 + $4 + $7 + $8, $5 + $6 }' /proc/stat)" \
        "$(awk -v d=${D##*/} '$3 == d { print $4, $8 }' /proc/diskstats)" "$(free -m | awk '/^Mem:/ { print $3 }')"
    sleep 30
done) > $S/$T.mon &
W=$!
(while true; do
    sleep 600
    gcloud storage cp -q $S/$T.log $S/$T.mon $S/$T.arrows $O/ 2> /dev/null || true
done) &
U=$!
# [4] run
X=0
if [ $ARM != ours ]; then
    if [ $DATA != sift ]; then
        until gcloud storage ls $B/$DATA/$BOOK > /dev/null 2>&1; do sleep 60; done
        gcloud storage cp $B/$DATA/$BOOK $S/
        curl -LsSf https://astral.sh/uv/install.sh | sh
        (cd $E && $HOME/.local/bin/uv run "0014/pqconvert.py" $S/$BOOK $S/$PIVOTS $DIM $BAND)
        gcloud storage cp -q $S/$PIVOTS $B/$DATA/ || true
    else
        gcloud storage cp $B/$DATA/$PIVOTS $S/
    fi
    knobs=(every=10 threads=64 swidth=2 iwidth=2 dwidth=2 reuse=1)
    extra=()
    [ $ARM$MODE = odinchurn ] && extra=(budget=1500000)
    mkdir -p $S/odin
    env PIPEANN_FIXED_BEAM_WIDTH=1 $E/build/${ARM}mix $LABEL $S/$POINTS $S/$GRAPH $S/$QUERY $S/$PIVOTS $S/odin/odin \
        $S/${T}build.arrows mode=grow batches=0 beam=100 repeat=1 "${knobs[@]}" > $S/${T}build.log 2>&1 || X=$?
    gcloud storage cp -q $S/${T}build.log $O/ || true
    dd if=/dev/zero of=$S/odin/odin_disk.index bs=64M count=8192 oflag=append,direct conv=notrunc status=none
    echo "PHASE prefilled $(date +%T)"
    env PIPEANN_FIXED_BEAM_WIDTH=1 PIPEANN_INSERT_FNCT=1 PIPEANN_POLL_SPINS=1 GLIBC_TUNABLES=$M $E/build/${ARM}mix \
        $LABEL $S/$POINTS $S/$GRAPH $S/$QUERY $S/$PIVOTS $S/odin/odin $S/$T.arrows mode=$MODE batches=100 \
        step=1000000 "${knobs[@]}" "${extra[@]}" > $S/$T.log 2>&1 || X=$?
else
    knobs=(mode=$MODE batches=100 every=10 step=1000000 stable=4 stall=8 scale=1.05 blunt=false width=2 band=$BAND)
    if [ $DATA = sift ]; then
        mkdir -p $S/snap
        gcloud storage cp $B/snap/sift100m-z3/index $B/snap/sift100m-z3/snap.bin $S/snap/
        $E/target/release/instantmix $LABEL $S/$POINTS $S/$GRAPH $S/$QUERY - $S/snap/index $S/$T.arrows "${knobs[@]}" \
            open=$S/snap/snap.bin start=100000000 > $S/$T.log 2>&1 || X=$?
    else
        mkdir -p $S/ours
        P=$B/$DATA/
        [ $MODE = churn ] && P=$O/
        (until [ -s $S/$BOOK ]; do sleep 30; done; sleep 10
            gcloud storage cp -q $S/$BOOK $P || true) &
        $E/target/release/instantmix $LABEL $S/$POINTS $S/$GRAPH $S/$QUERY - $S/ours/index $S/$T.arrows "${knobs[@]}" \
            book=$S/$BOOK > $S/$T.log 2>&1 || X=$?
    fi
fi
echo "PHASE $T rc $X $(date +%T)"
# [5] stop the monitor and the uploader, copy the results
kill $W $U || true
gcloud storage cp $S/$T.log $S/$T.mon $S/$T.arrows $O/
for core in $S/core.*; do
    [ -e "$core" ] || continue
    gcloud storage cp $core $O/ || true
    gcloud storage cp $E/build/${ARM}mix $E/target/release/instantmix $O/ 2> /dev/null || true
done
exit $X
