#!/bin/bash
# gcprun.sh <eval commit> <fnct commit> <name> <grow|churn> [knobs...]: #028's run of ours (instantmix) on the SIFT 899M
# base, 100 batches of 1M inserts (churn also deleting the oldest 1M), on a gcloud Batch z3-highmem-16-highlssd VM:
# built from this repo's `git archive` and fnct's in the bucket; <bucket>/sift1b/'s 1B rows, queries and 899M bidir
# graph on the local SSD, the graph waited for until it lands, so the VM may be started before the build ends; the log, the MON samples (epoch, busy and idle jiffies, SSD reads and writes completed, used MB)
# and the arrows to runs/<name>/ every 10 min and at the end, then #028/analyze.py scores the run (M 899) if the bucket
# holds the ground truth, its CSVs to runs/<name>/ too
set -euo pipefail
export HOME=/root
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
B=${BUCKET:?set BUCKET to a GCS bucket holding the inputs and the sources}
S=/mnt/lssd
E=/var/tmp/eval
D=$(readlink -f /dev/disk/by-id/google-local-nvme-ssd-0)
N=$3
MODE=$4
T=ours${MODE}_sift1b
KNOBS=(mode=$MODE batches=100 every=10 step=1000000 stable=4 stall=8 scale=1.05 width=2 band=8 degree=64 limit=32
    memcap=20000 blunt=false "${@:5}")
# [1] prepare: the local SSD, the toolchain, the sources, the build, uv, the inputs
mkfs.ext4 -q -F $D
mkdir -p $S
mount $D $S
systemctl stop unattended-upgrades apt-daily.timer apt-daily-upgrade.timer || true
sysctl -w fs.aio-max-nr=1048576
ulimit -n 1048576
lscpu > $S/$T.cpu
gcloud storage cp $S/$T.cpu $B/runs/$N/
(gcloud storage cp $B/sift1b/sift1b.1000M.raw $B/sift1b/sift1b.query $S/
    until gcloud storage ls $B/sift1b/sift1b.899M.flat.bidir > /dev/null 2>&1; do sleep 60; done
    gcloud storage cp $B/sift1b/sift1b.899M.flat.bidir $S/) &
C=$!
apt-get -o DPkg::Lock::Timeout=600 -qq update
DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -y -qq --no-install-recommends install \
    build-essential cmake libaio-dev liburing-dev curl ca-certificates
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p $E/fnct
gcloud storage cat $B/src/eval-$1.tar | tar x -C $E
gcloud storage cat $B/src/fnct-$2.tar | tar x -C $E/fnct
cd $E
cargo build --release --bin instantmix
wait $C
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
target/release/instantmix sift $S/sift1b.1000M.raw $S/sift1b.899M.flat.bidir $S/sift1b.query - $S/index/index \
    $S/$T.arrows "${KNOBS[@]}" book=$S/$T.book > $S/$T.log 2>&1 &
R=$!
# [3] copy the log, the samples and the arrows every 10 min while it runs, the codebook once written
(
    n=0
    while kill -0 $R 2>/dev/null; do
        sleep 10
        n=$(( n + 1 ))
        if (( n % 60 == 0 )); then gcloud storage cp -q $S/$T.log $S/$T.mon $S/$T.arrows $B/runs/$N/ || true; fi
        if [ -s $S/$T.book ] && [ ! -e $S/book.sent ]; then
            sleep 10; gcloud storage cp -q $S/$T.book $B/runs/$N/ && touch $S/book.sent || true
        fi
    done
) &
U=$!
# [4] wait for the run, copy the results, score them
X=0
wait $R || X=$?
wait $U || true
kill $W || true
gcloud storage cp $S/$T.log $S/$T.mon $S/$T.arrows $B/runs/$N/
gcloud storage cp $B/sift1b/sift1b.gt.arrows $S/ || true
[ -s $S/sift1b.gt.arrows ] && uv run --script "0028/analyze.py" $S/$T.arrows $S/sift1b.gt.arrows 899 \
    $S/recall_$T.csv $S/tp_$T.csv > $S/an_$T.log 2>&1 || true
touch $S/an_$T.log
gcloud storage cp $S/an_$T.log $B/runs/$N/
gcloud storage cp $S/recall_$T.csv $S/tp_$T.csv $B/runs/$N/ || true
exit $X
