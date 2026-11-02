#!/bin/bash
# gcpbuild.sh <eval commit> <fnct commit> <name> [pipnn flags...]: #028's SIFT 899M base on a gcloud Batch
# c4-highmem-288-lssd VM: BigANN's base.1B.u8bin and query.public.10K.u8bin from dl.fbaipublicfiles.com onto the local
# SSDs (RAID0), checked against a reference copy by the md5 of the first and last 10M rows and of the queries; the headerless
# 1B rows and queries to <bucket>/sift1b/ for the runs; the first 899M rows as a u8bin to PiPNN 443a328 (mirage left out
# of its CMake: it needs BLAS and is unused) with the README's flags at alpha 1.2 plus [pipnn flags...], then
# #002/flatgraph.py and bidir flat 64, the bidir graph to <bucket>/sift1b/sift1b.899M.flat.bidir; a MON line every 30 s
# (epoch, busy and idle jiffies, used MB) and the log to <bucket>/runs/<name>/ every 2 min and at the end
set -euo pipefail
export HOME=/root
export PATH=$HOME/.cargo/bin:$HOME/.local/bin:$PATH
B=${BUCKET:?set BUCKET to a GCS bucket holding the inputs and the sources}
O=$B/runs/$3
S=/mnt/lssd
E=/var/tmp/eval
P=$S/PiPNN
U=https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann
BASE=899000000
EXTRA=("${@:4}")
L=$S/build.log
mkdir -p $S
# [1] the local SSDs as one RAID0, the monitor and the uploader
mapfile -t DISKS < <(ls /dev/disk/by-id/google-local-nvme-ssd-* | grep -v part)
apt-get -o DPkg::Lock::Timeout=600 -qq update
DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -y -qq --no-install-recommends install \
    build-essential cmake git mdadm aria2 time curl ca-certificates python3
if [ ${#DISKS[@]} -gt 1 ]; then
    mdadm --create /dev/md0 --level=0 --raid-devices=${#DISKS[@]} --run "${DISKS[@]}"
    mkfs.ext4 -q -F /dev/md0
    mount /dev/md0 $S
else
    mkfs.ext4 -q -F "${DISKS[0]}"
    mount "${DISKS[0]}" $S
fi
systemctl stop unattended-upgrades apt-daily.timer apt-daily-upgrade.timer || true
exec > >(tee -a $L) 2>&1
echo "PHASE disks ${#DISKS[@]} $(df -h $S | tail -1) $(date +%T)"
lscpu > $S/lscpu.txt
(while true; do
    echo "MON $(date +%s) $(awk '/^cpu / { print $2 + $3 + $4 + $7 + $8, $5 + $6 }' /proc/stat) $(free -m | awk '/^Mem:/ { print $3 }')"
    sleep 30
done) > $S/build.mon &
W=$!
(while true; do
    sleep 120
    gcloud storage cp -q $L $S/build.mon $O/ 2> /dev/null || true
done) &
V=$!
# [2] the inputs in the background, the toolchains and sources, PiPNN and bidir built
(aria2c -q -x 16 -s 16 -k 256M --file-allocation=none -d $S -o base.1B.u8bin $U/base.1B.u8bin
    aria2c -q -d $S -o query.u8bin $U/query.public.10K.u8bin
    echo "PHASE downloaded $(date +%T)") &
C=$!
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal
curl -LsSf https://astral.sh/uv/install.sh | sh
mkdir -p $E/fnct
gcloud storage cat $B/src/eval-$1.tar | tar x -C $E
gcloud storage cat $B/src/fnct-$2.tar | tar x -C $E/fnct
(cd $E && cargo build -q --release --bin bidir)
git clone -q https://github.com/ParAlg/PiPNN.git $P
git -C $P checkout -q 443a328
git -C $P submodule update -q --init
sed -i 's|^add_subdirectory(external/mirage)|#&|' $P/CMakeLists.txt
sed -i 's|^add_subdirectory(mirage)|#&|' $P/algorithms/CMakeLists.txt
cmake -S $P -B $P/build -DCMAKE_BUILD_TYPE=Release > $S/cmake.log 2>&1
cmake --build $P/build --target neighbors-pipnn_UINT8_T_EUCLIDEAN -j 64 > $S/make.log 2>&1
echo "PHASE built $(date +%T)"
wait $C
# [3] the rows: headerless 1B and queries checked against a reference copy, uploaded; the 899M u8bin for PiPNN
tail -c +9 $S/base.1B.u8bin > $S/sift1b.1000M.raw
rm $S/base.1B.u8bin
tail -c +9 $S/query.u8bin > $S/sift1b.query
H=$(head -c 1280000000 $S/sift1b.1000M.raw | md5sum | cut -c1-32)
T=$(tail -c 1280000000 $S/sift1b.1000M.raw | md5sum | cut -c1-32)
Q=$(md5sum < $S/sift1b.query | cut -c1-32)
echo "PHASE md5 head $H tail $T query $Q size $(stat -c %s $S/sift1b.1000M.raw) $(date +%T)"
[ "$H" = e9f57c9b08caf2fa76be00d1b44bf798 ] && [ "$T" = bcfb95f53e12da0f7d16a371ca158f17 ] \
    && [ "$Q" = 988af2f5bab1311c880e55e5f1da3822 ] || { echo "PHASE md5 mismatch"; exit 3; }
gcloud storage cp -q $S/sift1b.1000M.raw $S/sift1b.query $B/sift1b/ &
X=$!
python3 -c "import struct,sys; sys.stdout.buffer.write(struct.pack('<II',$BASE,128))" > $S/sift1b.899M.u8bin
head -c $(( BASE * 128 )) $S/sift1b.1000M.raw >> $S/sift1b.899M.u8bin
echo "PHASE cropped $(date +%T)"
# [4] PiPNN, flatgraph.py, bidir, the bidir graph uploaded
/usr/bin/time -v $P/build/algorithms/PipNN/neighbors-pipnn_UINT8_T_EUCLIDEAN -base_path $S/sift1b.899M.u8bin \
    -file_type bin -alpha 1.2 -num_clusters 1 -fanout 30 -fanout_scheme 10,3 -cluster_size 1024 -mst_deg 2 \
    -multi_pivot -prune -prune_all -prune_degree 64 -top_level_leaders 1000 -fraction_leaders 0.005 -hash_bits 12 \
    -leaf_method EigenKNN "${EXTRA[@]}" -graph_outfile $S/sift1b.899M.graph.orig > $S/pipnn.log 2>&1
gcloud storage cp -q $S/pipnn.log $O/
echo "PHASE pipnn $(date +%T)"
rm $S/sift1b.899M.u8bin
(cd $E && /usr/bin/time -v uv run "0002/flatgraph.py" $S/sift1b.899M.graph.orig $S/sift1b.899M.flat.orig) \
    > $S/flatgraph.log 2>&1
echo "PHASE flatgraph $(date +%T)"
/usr/bin/time -v $E/target/release/bidir flat 64 $S/sift1b.899M.flat.orig $S/sift1b.899M.flat.bidir > $S/bidir.log 2>&1
echo "PHASE bidir $(date +%T)"
gcloud storage cp -q $S/sift1b.899M.flat.bidir $B/sift1b/
wait $X
echo "PHASE uploaded $(date +%T)"
kill $W $V || true
gcloud storage cp -q $L $S/build.mon $S/lscpu.txt $S/cmake.log $S/make.log $S/pipnn.log $S/flatgraph.log $S/bidir.log $O/
