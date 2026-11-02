#!/bin/bash
# gcprun.sh <eval commit> <pipeann commit> <fnct commit> <fresh commit> <ds> <system> <name> [knobs...]: #024's run of
# one system (odin, or ours as instantmix with the knobs) on a 1B wavy file's first 201M rows, a 1M base grown and then
# churned per the knobs, on a gcloud Batch z3-highmem-16-highlssd VM: built from this repo's `git archive` and the
# submodules' at the given commits in the bucket with the same build options, the inputs from $BUCKET/<ds>/ on
# the local SSD; the log, the MON samples and the arrows go to runs/<name>/ every 10 min and at the end, then analyze.py
# scores the run (M 1, G the knob grow) and its CSVs go there too; the knobs rows=, base= and block= (not passed on) name
# the stream <ds>.<rows>.raw (201M by default) and the base <ds>.<base>.flat.bidir and .pq_pivots.bin (1M), and set the
# ground truth's BLOCK (1000000); the ground truth is copied and scored only if the bucket holds it
set -euo pipefail
export HOME=/root
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
B=${BUCKET:?set BUCKET to a GCS bucket holding the inputs and the sources}
S=/mnt/lssd
E=/var/tmp/eval
D=$(readlink -f /dev/disk/by-id/google-local-nvme-ssd-0)
DS=$5
SYS=$6
N=$7
T=${SYS}_${DS}
case $DS in msturing*) LABEL=msturing ;; spacev*) LABEL=spacev ;; sift*) LABEL=sift ;; deep*) LABEL=deep ;; *) LABEL=${DS%%w1b} ;; esac
M=glibc.malloc.mmap_threshold=33554432:glibc.malloc.trim_threshold=17179869184:glibc.malloc.top_pad=268435456
GROW=0
ROWS=201M
BASE=1M
BLOCK=1000000
STEP=1000000
KNOBS=()
for k in "${@:8}"; do
    case $k in
        rows=*) ROWS=${k#rows=} ;;
        base=*) BASE=${k#base=} ;;
        block=*) BLOCK=${k#block=} ;;
        *) KNOBS+=("$k"); case $k in grow=*) GROW=${k#grow=} ;; step=*) STEP=${k#step=} ;; esac ;;
    esac
done
export BLOCK
# [1] prepare: the local SSD, the toolchain, OpenBLAS as the cblas and an MKL header stub, the sources, the build, uv
mkfs.ext4 -q -F $D
mkdir -p $S
mount $D $S
systemctl stop unattended-upgrades apt-daily.timer apt-daily-upgrade.timer || true
sysctl -w fs.aio-max-nr=1048576
ulimit -n 1048576
lscpu > $S/$T.cpu
gcloud storage cp $S/$T.cpu $B/runs/$N/
apt-get -o DPkg::Lock::Timeout=600 -qq update
DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -y -qq --no-install-recommends install \
    build-essential cmake libboost-dev libaio-dev liburing-dev libopenblas-dev libeigen3-dev curl ca-certificates
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
cd $E
if [ $SYS = odin ]; then
    (cd $E/pipeann/third_party/liburing && ./configure && make -C src -j 16)
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON_INTERFACE=OFF -DFRESHONLY=OFF -DFRESH_IO=uring \
        -DIO_ENGINE=aio -DPIPEANN_BG_IO_THREADS=1 -DPIPEANN_DENSE_LOCKS=OFF -DPIPE_TRACE=OFF -DUSE_TCMALLOC=OFF
    cmake --build build --target odinmix -j 16
else
    cargo build --release --bin instantmix
fi
gcloud storage cp $B/$DS/$DS.$ROWS.raw $B/$DS/$DS.query $B/$DS/$DS.$BASE.flat.bidir $B/$DS/$DS.$BASE.pq_pivots.bin $S/
gcloud storage cp $B/$DS/$DS.gt.arrows $S/ || true
mkdir -p $S/index
# [2] run, with the MON sampler beside it
(
    K=$(basename "$D")
    while true; do
        echo "MON $(date +%s) $(awk '/^cpu / { print $2 + $3 + $4 + $7 + $8, $5 + $6 }' /proc/stat)" \
            "$(awk -v d=$K '$3 == d { print $4, $8 }' /proc/diskstats)" "$(free -m | awk '/^Mem:/ { print $3 }')"
        sleep 30
    done
) > $S/$T.mon &
W=$!
if [ $SYS = odin ]; then
    env PIPEANN_FIXED_BEAM_WIDTH=1 build/odinmix $LABEL $S/$DS.$ROWS.raw $S/$DS.$BASE.flat.bidir $S/$DS.query \
        $S/$DS.$BASE.pq_pivots.bin $S/index/odin $S/${T}build.arrows mode=grow batches=0 step=$STEP beam=100 repeat=1 \
        every=1 threads=64 swidth=2 iwidth=2 dwidth=2 reuse=1 > $S/${T}build.log 2>&1
    dd if=/dev/zero of=$S/index/odin_disk.index bs=64M count=2048 oflag=append,direct conv=notrunc status=none
    env PIPEANN_FIXED_BEAM_WIDTH=1 PIPEANN_INSERT_FNCT=1 PIPEANN_POLL_SPINS=1 GLIBC_TUNABLES=$M build/odinmix $LABEL \
        $S/$DS.$ROWS.raw $S/$DS.$BASE.flat.bidir $S/$DS.query $S/$DS.$BASE.pq_pivots.bin $S/index/odin $S/$T.arrows "${KNOBS[@]}" \
        > $S/$T.log 2>&1 &
else
    target/release/instantmix $LABEL $S/$DS.$ROWS.raw $S/$DS.$BASE.flat.bidir $S/$DS.query - $S/index/index \
        $S/$T.arrows "${KNOBS[@]}" book=$S/$T.book > $S/$T.log 2>&1 &
fi
R=$!
# [3] copy the log, the samples and the arrows every 10 min while it runs
(
    n=0
    while kill -0 $R 2>/dev/null; do
        sleep 10
        n=$(( n + 1 ))
        if (( n % 60 == 0 )); then gcloud storage cp $S/$T.log $S/$T.mon $S/$T.arrows $B/runs/$N/ || true; fi
    done
) &
U=$!
# [4] wait for the run, copy the results, score them
X=0
wait $R || X=$?
wait $U || true
kill $W || true
gcloud storage cp $S/$T.log $S/$T.mon $S/$T.arrows $B/runs/$N/
if [ $SYS = odin ]; then gcloud storage cp $S/${T}build.log $B/runs/$N/; fi
[ -s $S/$DS.gt.arrows ] && uv run "0015/analyze.py" $S/$T.arrows $S/$DS.gt.arrows 1 $S/rc_$T.csv $S/tp_$T.csv $GROW \
    > $S/an_$T.log 2>&1 || true
touch $S/an_$T.log
gcloud storage cp $S/an_$T.log $B/runs/$N/
gcloud storage cp $S/rc_$T.csv $S/tp_$T.csv $B/runs/$N/ || true
exit $X
