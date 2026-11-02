#!/bin/bash
set -e
cd "$(dirname "$0")"
: "${BUCKET:?}" "${DATA:?}"
git submodule update --init --recursive .
export E=$PWD COMMIT=$(git rev-parse --short HEAD)
git archive --format=tar HEAD . | gcloud storage cp - $BUCKET/src/eval-$COMMIT.tar
tar cf - -T /dev/null | gcloud storage cp - $BUCKET/src/fresh-2c185ab.tar
git -C fnct archive --format=tar HEAD | gcloud storage cp - $BUCKET/src/fnct-578c7a3.tar
git -C pipeann archive --format=tar HEAD | gcloud storage cp - $BUCKET/src/pipeann-d4e7da2.tar
cargo build --release --bin crop --bin gtbrute --bin bidir --bin instantmix
make -C parlayann/algorithms/vamana
mkdir -p $DATA
cd $DATA
bash $E/download.sh spacev spacev_q
uv run $E/0024/wave.py spacev.1B.100.i8.bin i8 100 201000000 spacevw1b.201M.raw
cp spacev.29316.100.i8.query spacevw1b.query
$E/target/release/crop spacevw1b.201M.raw spacevw1b.1M.bin 1000000 100 1
$E/parlayann/algorithms/vamana/neighbors -R 64 -L 128 -alpha 1.2 -num_passes 2 -data_type int8 -dist_func Euclidian -base_path spacevw1b.1M.bin -graph_outfile spacevw1b.1M.graph.orig
uv run $E/0002/flatgraph.py spacevw1b.1M.graph.orig spacevw1b.1M.flat.orig
$E/target/release/bidir flat 64 spacevw1b.1M.flat.orig spacevw1b.1M.flat.bidir
$E/target/release/gtbrute i8 100 spacevw1b.201M.raw spacevw1b.query spacevw1b.gt.arrows spacevw1b.scratch
mkdir -p paper
$E/target/release/instantmix spacev spacevw1b.201M.raw spacevw1b.1M.flat.bidir spacevw1b.query - paper/index book.arrows stable=4 stall=8 scale=1.05 width=2 band=8 degree=64 limit=32 book=spacevw1b.1M.book mode=churn batches=0 step=1000000
uv run $E/0014/pqconvert.py spacevw1b.1M.book spacevw1b.1M.pq_pivots.bin 100 8
gcloud storage cp spacevw1b.201M.raw spacevw1b.query spacevw1b.1M.flat.bidir spacevw1b.1M.pq_pivots.bin spacevw1b.gt.arrows $BUCKET/spacevw1b/
cd $E
K="mode=churn grow=99 batches=199 step=1000000 every=10"
U="stable=4 stall=8 scale=1.05 width=2 band=8 degree=64 limit=32 memcap=20000"
A="d4e7da2 578c7a3 2c185ab spacevw1b"
sed -e "s|BUCKETURI|$BUCKET|g" -e "s|ARGS|$A water shift-water $K $U blunt=false|g" -e "s/COMMIT/$COMMIT/g" -e "s/MAXRUN/172800s/g" 0024/gcprun.json > job.json
gcloud batch jobs submit shift-water --location=us-central1 --config=job.json
sed -e "s|BUCKETURI|$BUCKET|g" -e "s|ARGS|$A blunt shift-blunt $K $U blunt=true|g" -e "s/COMMIT/$COMMIT/g" -e "s/MAXRUN/172800s/g" 0024/gcprun.json > job.json
gcloud batch jobs submit shift-blunt --location=us-central1 --config=job.json
sed -e "s|BUCKETURI|$BUCKET|g" -e "s|ARGS|$A odin shift-odin $K threads=64 swidth=2 iwidth=2 dwidth=2 reuse=1 budget=1500000|g" -e "s/COMMIT/$COMMIT/g" -e "s/MAXRUN/345600s/g" 0024/gcprun.json > job.json
gcloud batch jobs submit shift-odin --location=us-central1 --config=job.json
mkdir -p runs
cd runs
gcloud storage cp $BUCKET/spacevw1b/spacevw1b.gt.arrows .
for job in water:ours blunt:oursnops odin:odin; do
  S=${job%%:*}
  N=${job#*:}
  gcloud storage cp $BUCKET/runs/shift-$S/${S}_spacevw1b.arrows .
  uv run $E/0015/analyze.py ${S}_spacevw1b.arrows spacevw1b.gt.arrows 1 recall_${N}_shift.csv tp_${N}_shift.csv
done
cd $E
uv run tables.py runs .
uv run plot.py
