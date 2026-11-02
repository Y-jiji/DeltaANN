#!/bin/bash
set -e
cd "$(dirname "$0")"
: "${BUCKET:?}" "${DATA:?}"
git submodule update --init --recursive .
export E=$PWD COMMIT=$(git rev-parse --short HEAD)
git archive --format=tar HEAD . | gcloud storage cp - $BUCKET/src/eval-$COMMIT.tar
tar cf - -T /dev/null | gcloud storage cp - $BUCKET/src/fresh-2c185ab.tar
git -C fnct archive --format=tar HEAD | gcloud storage cp - $BUCKET/src/fnct-bf5d938.tar
cargo build --release --bin crop --bin gtbrute --bin bidir --bin instantmix
make -C parlayann/algorithms/vamana
mkdir -p $DATA
cd $DATA
bash $E/download.sh sift sift_q
head -c $((201000000 * 128)) sift.1B.128.u8.bin > sift.201M.128.u8.bin
$E/target/release/crop sift.1B.128.u8.bin sift.100M.128.u8bin 100000000 128 1
$E/parlayann/algorithms/vamana/neighbors -R 64 -L 128 -alpha 1.2 -num_passes 2 -data_type uint8 -dist_func Euclidian -base_path sift.100M.128.u8bin -graph_outfile sift.100M.graph.orig
uv run $E/0002/flatgraph.py sift.100M.graph.orig sift.100M.flat.orig
$E/target/release/bidir flat 64 sift.100M.flat.orig sift.100M.flat.bidir
$E/target/release/gtbrute u8 128 sift.201M.128.u8.bin sift.10K.128.u8.query sift.gt.arrows sift.scratch
mkdir -p snap
$E/target/release/instantmix sift sift.201M.128.u8.bin sift.100M.flat.bidir sift.10K.128.u8.query - snap/index load.arrows mode=grow batches=0 step=1000000 stable=4 stall=8 scale=1.05 blunt=false width=2 band=8 book=sift.100M.instwater.book save=snap/snap.bin
uv run $E/0014/pqconvert.py sift.100M.instwater.book sift.100M.instwater.pq_pivots.bin 128 8
gcloud storage cp sift.201M.128.u8.bin sift.100M.flat.bidir sift.10K.128.u8.query sift.gt.arrows sift.100M.instwater.book sift.100M.instwater.pq_pivots.bin $BUCKET/sift/
gcloud storage cp snap/index snap/snap.bin $BUCKET/snap/sift100m-z3/
cd $E
sed -e "s|BUCKETURI|$BUCKET|g" -e "s/COMMIT/$COMMIT/g" -e "s/NAME/study-0/g" -e "s/VARIANTS/full:10:90 c15:10:120:limit=43 c3:10:90:limit=21 c4:10:90:limit=16/g" -e "s/CAP/25200/g" 0029/gcpstudy.json > job.json
gcloud batch jobs submit study-0 --location=us-central1 --config=job.json
sed -e "s|BUCKETURI|$BUCKET|g" -e "s/COMMIT/$COMMIT/g" -e "s/NAME/study-1/g" -e "s/VARIANTS/b100:5:50:scale=1.0 b110:5:50:scale=1.1 b120:5:50:scale=1.2/g" -e "s/CAP/9000/g" 0029/gcpstudy.json > job.json
gcloud batch jobs submit study-1 --location=us-central1 --config=job.json
sed -e "s|BUCKETURI|$BUCKET|g" -e "s/COMMIT/$COMMIT/g" -e "s/NAME/study-2/g" -e "s/VARIANTS/nt1:2:90:ntask=1 nt2:2:50:ntask=2 nt4:2:35:ntask=4 nt16:2:30:ntask=16/g" -e "s/CAP/13500/g" 0029/gcpstudy.json > job.json
gcloud batch jobs submit study-2 --location=us-central1 --config=job.json
mkdir -p runs
cd runs
gcloud storage cp $BUCKET/sift/sift.gt.arrows .
for tag in full c15 c3 c4; do
  gcloud storage cp $BUCKET/runs/study-0/$tag/$tag.arrows .
  uv run $E/0029/analyze.py $tag.arrows sift.gt.arrows 100 recall_$tag.csv tput_$tag.csv
  uv run $E/0029/counters.py $tag.arrows counters_$tag.csv
done
for tag in b100 b110 b120; do
  gcloud storage cp $BUCKET/runs/study-1/$tag/$tag.arrows .
  uv run $E/0029/analyze.py $tag.arrows sift.gt.arrows 100 recall_$tag.csv tput_$tag.csv
  uv run $E/0029/counters.py $tag.arrows counters_$tag.csv
done
for tag in nt1 nt2 nt4 nt16; do
  gcloud storage cp $BUCKET/runs/study-2/$tag/$tag.arrows .
  uv run $E/0029/analyze.py $tag.arrows sift.gt.arrows 100 recall_$tag.csv tput_$tag.csv
  uv run $E/0029/counters.py $tag.arrows counters_$tag.csv
done
cp $E/0029/summary.py .
uv run summary.py full c15 c3 c4 b100 b110 b120 nt1 nt2 nt4 nt16 batches.csv quality.csv
cd $E
uv run tables.py runs .
uv run plot.py
