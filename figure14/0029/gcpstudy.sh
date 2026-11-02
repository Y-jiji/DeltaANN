#!/bin/bash
# gcpstudy.sh <commit> <name> <variant>...: #029's SIFT 100M churn study on one gcloud Batch z3-highmem-8-highlssd VM,
# the variants in turn, each from a fresh copy of #015's sift100m-z3 snapshot as gcpfull.sh's ours churn: instantmix
# against fnct bf5d938 with 0029/study.patch, 1:1:1 churn of 1M:1M:1M batches, gcpfull.sh's knobs (stable 4, stall 8,
# scale 1.05, blunt false, width 2, band 8) then the variant's; a variant `tag:batches:minutes[:a=b...]`, its
# assignments knobs except REPAIR and LIMIT, set in the environment, its run killed after minutes (the trace keeps every finished
# stage); the queries swept after the load, every 5 batches and after the last; a MON line every 30 s (epoch, busy and
# idle jiffies, SSD reads and writes completed, used MB); each variant's log, MON and arrows go to runs/<name>/<tag>/
# every 10 minutes and at its end
set -euo pipefail
export HOME=/root
export PATH=$HOME/.cargo/bin:$PATH
B=${BUCKET:?set BUCKET to a GCS bucket holding the inputs and the sources}
E=/var/tmp/eval
S=/mnt/lssd
FNCT=bf5d938
O=$B/runs/$2
D=$(readlink -f /dev/disk/by-id/google-local-nvme-ssd-0)
POINTS=sift.201M.128.u8.bin
GRAPH=sift.100M.flat.bidir
QUERY=sift.10K.128.u8.query
# [1] prepare: the local SSD, the inputs and the snapshot in the background, the toolchain, the patched sources
mkfs.ext4 -q -F $D
mkdir -p $S
mount $D $S
mkdir -p $S/snap0
systemctl stop unattended-upgrades apt-daily.timer apt-daily-upgrade.timer || true
sysctl -w fs.aio-max-nr=1048576
sysctl -w kernel.core_pattern=$S/core.%e.%p
ulimit -n 1048576
ulimit -c unlimited
(gcloud storage cp $B/sift/$POINTS $B/sift/$GRAPH $B/sift/$QUERY $S/
    gcloud storage cp $B/snap/sift100m-z3/index $B/snap/sift100m-z3/snap.bin $S/snap0/) &
C=$!
apt-get -o DPkg::Lock::Timeout=600 -qq update
DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -y -qq --no-install-recommends install \
    build-essential cmake libboost-dev libaio-dev liburing-dev libopenblas-dev libeigen3-dev curl ca-certificates patch
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
mkdir -p $E/fnct
tar xf /var/tmp/boot/eval-$1.tar -C $E
gcloud storage cat $B/src/fnct-$FNCT.tar | tar x -C $E/fnct
# 0029/study.patch adds the `full` and `sdc` counters and the REPAIR setting to the private library; it is not published here
[ -f $E/0029/study.patch ] && (cd $E/fnct && patch -p1 < $E/0029/study.patch)
cd $E
cargo build --release --bin instantmix
wait $C
echo "PHASE built $(date +%T)"
# [2] run: each variant from a fresh snapshot copy, its own MON and uploader
X=0
for V in "${@:3}"; do
    IFS=: read -ra F <<< "$V"
    T=${F[0]}
    envs=()
    knobs=(mode=churn batches=${F[1]} every=5 step=1000000 stable=4 stall=8 scale=1.05 blunt=false width=2 band=8)
    for a in "${F[@]:3}"; do
        case $a in
            REPAIR=*|LIMIT=*) envs+=("$a") ;;
            *) knobs+=("$a") ;;
        esac
    done
    rm -rf $S/snap
    cp -r $S/snap0 $S/snap
    (while true; do
        echo "MON $(date +%s) $(awk '/^cpu / { print $2 + $3 + $4 + $7 + $8, $5 + $6 }' /proc/stat)" \
            "$(awk -v d=${D##*/} '$3 == d { print $4, $8 }' /proc/diskstats)" "$(free -m | awk '/^Mem:/ { print $3 }')"
        sleep 30
    done) > $S/$T.mon &
    W=$!
    (while true; do
        sleep 600
        gcloud storage cp -q $S/$T.log $S/$T.mon $S/$T.arrows $O/$T/ 2> /dev/null || true
    done) &
    U=$!
    echo "PHASE $T start $(date +%T) ${envs[*]} ${knobs[*]}"
    R=0
    env "${envs[@]}" timeout ${F[2]}m $E/target/release/instantmix sift $S/$POINTS $S/$GRAPH $S/$QUERY - \
        $S/snap/index $S/$T.arrows "${knobs[@]}" open=$S/snap/snap.bin start=100000000 > $S/$T.log 2>&1 || R=$?
    echo "PHASE $T rc $R $(date +%T)"
    kill $W $U || true
    gcloud storage cp $S/$T.log $S/$T.mon $S/$T.arrows $O/$T/ || true
    [ $R = 0 ] || X=$R
done
# [3] cores, if any, with the binary
for core in $S/core.*; do
    [ -e "$core" ] || continue
    gcloud storage cp $core $E/target/release/instantmix $O/ || true
done
exit $X
