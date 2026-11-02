#!/bin/bash
# gcpodin.sh <eval commit> <pipeann commit> <fnct commit> <fresh commit> <name> <grow|churn> [cpus] [knobs...]: #028's
# run of OdinANN (odinmix, #024's build options and search and insert settings) on the SIFT 899M base, 100 batches of
# 1M inserts (churn also deleting the oldest 1M), on a gcloud Batch VM: pipeann with #028/pq64.patch, whose 64-bit
# multiply lets a 16-byte PQ code be read past id 268,435,456; the pivots from run028's codebook by #014/pqconvert.py,
# so OdinANN and ours code rows alike; the index built on the first local SSD, then 512 GiB of zeros appended; the run
# under `taskset -c 0-<cpus-1>` if cpus is not 0, the build on every vCPU; a MON line every 30 s (epoch, busy and idle
# jiffies, SSD reads and writes completed, used MB); the logs, MON and arrows to runs/<name>/ every 10 min and at the end,
# then #028/analyze.py scores the run (M 899) against the bucket's ground truth, its CSVs to runs/<name>/ too
set -euo pipefail
export HOME=/root
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
B=${BUCKET:?set BUCKET to a GCS bucket holding the inputs and the sources}
S=/mnt/lssd
E=/var/tmp/eval
D=$(readlink -f /dev/disk/by-id/google-local-nvme-ssd-0)
N=$5
MODE=$6
CPUS=${7:-0}
T=odin${MODE}_sift1b
BOOK=$B/runs/ours-grow/oursgrow_sift1b.book
M=glibc.malloc.mmap_threshold=33554432:glibc.malloc.trim_threshold=17179869184:glibc.malloc.top_pad=268435456
COMMON=(every=10 threads=64 swidth=2 iwidth=2 dwidth=2 reuse=1)
KNOBS=(mode=$MODE batches=100 step=1000000 "${COMMON[@]}" "${@:8}")
# [1] prepare: the local SSD, the toolchain, OpenBLAS as the cblas and an MKL header stub, the sources, the build, uv
mkfs.ext4 -q -F $D
mkdir -p $S
mount $D $S
systemctl stop unattended-upgrades apt-daily.timer apt-daily-upgrade.timer || true
sysctl -w fs.aio-max-nr=1048576
ulimit -n 1048576
lscpu > $S/$T.cpu
df -h $S >> $S/$T.cpu
gcloud storage cp $S/$T.cpu $B/runs/$N/
gcloud storage cp $B/sift1b/sift1b.1000M.raw $B/sift1b/sift1b.query $B/sift1b/sift1b.899M.flat.bidir $BOOK $S/ &
C=$!
apt-get -o DPkg::Lock::Timeout=600 -qq update
DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -y -qq --no-install-recommends install \
    build-essential cmake patch libboost-dev libaio-dev liburing-dev libopenblas-dev libeigen3-dev curl ca-certificates
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p /usr/local/include/mkl $E/pipeann $E/fnct $E/freshdiskann
touch /usr/local/include/mkl/mkl_cblas.h
ln -sf "$(find /usr/lib -name libopenblas.so | head -1)" /usr/local/lib/libcblas.so
export CPATH=/usr/include/eigen3:$(dirname "$(find /usr/include -name cblas.h | head -1)")
gcloud storage cat $B/src/eval-$1.tar | tar x -C $E
gcloud storage cat $B/src/pipeann-$2.tar | tar x -C $E/pipeann
gcloud storage cat $B/src/fnct-$3.tar | tar x -C $E/fnct
gcloud storage cat $B/src/fresh-$4.tar | tar x -C $E/freshdiskann
(cd $E/pipeann && patch -p1 < $E/0028/pq64.patch)
cd $E
(cd $E/pipeann/third_party/liburing && ./configure && make -C src -j 16)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_INTERFACE=OFF -DFRESHONLY=OFF -DFRESH_IO=uring \
    -DIO_ENGINE=aio -DPIPEANN_BG_IO_THREADS=1 -DPIPEANN_DENSE_LOCKS=OFF -DPIPE_TRACE=OFF -DUSE_TCMALLOC=OFF
cmake --build build --target odinmix -j 16
wait $C
uv run "0014/pqconvert.py" $S/oursgrow_sift1b.book $S/sift1b.pq_pivots.bin 128 8
mkdir -p $S/index
echo "PHASE prepared $(date +%T)"
# [2] the MON sampler, then the build and the zeros
(
    K=$(basename "$D")
    while true; do
        echo "MON $(date +%s) $(awk '/^cpu / { print $2 + $3 + $4 + $7 + $8, $5 + $6 }' /proc/stat)" \
            "$(awk -v d=$K '$3 == d { print $4, $8 }' /proc/diskstats)" "$(free -m | awk '/^Mem:/ { print $3 }')"
        sleep 30
    done
) > $S/$T.mon &
W=$!
(
    while true; do
        sleep 600
        gcloud storage cp -q $S/$T.mon $S/${T}build.log $S/$T.log $S/$T.arrows $B/runs/$N/ 2> /dev/null || true
    done
) &
U=$!
X=0
env PIPEANN_FIXED_BEAM_WIDTH=1 build/odinmix sift $S/sift1b.1000M.raw $S/sift1b.899M.flat.bidir $S/sift1b.query \
    $S/sift1b.pq_pivots.bin $S/index/odin $S/${T}build.arrows mode=grow batches=0 step=1000000 beam=100 repeat=1 \
    "${COMMON[@]}" > $S/${T}build.log 2>&1 || X=$?
echo "PHASE built rc=$X $(date +%T)"
gcloud storage cp -q $S/${T}build.log $B/runs/$N/ || true
if [ $X = 0 ]; then
    dd if=/dev/zero of=$S/index/odin_disk.index bs=64M count=8192 oflag=append,direct conv=notrunc status=none
    echo "PHASE prefilled $(df -h $S | tail -1) $(date +%T)"
    # [3] the run, on the same vCPU count as ours where the VM has more for its memory
    PIN=()
    [ "$CPUS" != 0 ] && PIN=(taskset -c "0-$((CPUS - 1))")
    env PIPEANN_FIXED_BEAM_WIDTH=1 PIPEANN_INSERT_FNCT=1 PIPEANN_POLL_SPINS=1 GLIBC_TUNABLES=$M "${PIN[@]}" build/odinmix \
        sift $S/sift1b.1000M.raw $S/sift1b.899M.flat.bidir $S/sift1b.query $S/sift1b.pq_pivots.bin $S/index/odin \
        $S/$T.arrows "${KNOBS[@]}" > $S/$T.log 2>&1 || X=$?
    echo "PHASE ran rc=$X $(date +%T)"
fi
# [4] stop the sampler and uploader, copy the results, score them
kill $W $U || true
gcloud storage cp $S/$T.mon $S/${T}build.log $B/runs/$N/
if [ -s $S/$T.arrows ]; then
    gcloud storage cp $S/$T.log $S/$T.arrows $B/runs/$N/
    gcloud storage cp $B/sift1b/sift1b.gt.arrows $S/
    uv run --script "0028/analyze.py" $S/$T.arrows $S/sift1b.gt.arrows 899 $S/recall_$T.csv $S/tp_$T.csv \
        > $S/an_$T.log 2>&1 || true
    touch $S/an_$T.log
    gcloud storage cp $S/an_$T.log $B/runs/$N/
    gcloud storage cp $S/recall_$T.csv $S/tp_$T.csv $B/runs/$N/ || true
fi
exit $X
