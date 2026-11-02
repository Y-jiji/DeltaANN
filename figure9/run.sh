#!/bin/bash
set -e
cd "$(dirname "$0")"
: "${BUCKET:?}" "${DATA:?}"
git submodule update --init --recursive .
export E=$PWD COMMIT=$(git rev-parse --short HEAD)
git archive --format=tar HEAD . | gcloud storage cp - $BUCKET/src/eval-$COMMIT.tar
tar cf - -T /dev/null | gcloud storage cp - $BUCKET/src/fresh-2c185ab.tar
git -C fnct archive --format=tar HEAD | gcloud storage cp - $BUCKET/src/fnct-bf5d938.tar
git -C pipeann archive --format=tar HEAD | gcloud storage cp - $BUCKET/src/pipeann-5147dab.tar
git -C parlayann archive --format=tar HEAD | gcloud storage cp - $BUCKET/src/parlayann-a48c6a3.tar
git -C parlayann/parlaylib archive --format=tar HEAD | gcloud storage cp - $BUCKET/src/parlaylib-c655b8f.tar
cargo build --release --bin crop --bin gtbrute --bin bidir --bin instantmix
cargo build --release --manifest-path 0019/Cargo.toml --target-dir target --bin reorder
make -C parlayann/algorithms/vamana
mkdir -p $DATA
cd $DATA
bash $E/download.sh sift sift_q deep deep_q spacev spacev_q msturing
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
head -c $((201000000 * 384)) deep.1B.96.f32.bin > deep.201M.96.f32.bin
$E/target/release/crop deep.1B.96.f32.bin deep.100M.96.fbin 100000000 96 4
gcloud storage cp deep.201M.96.f32.bin deep.100M.96.fbin deep.10K.96.f32.query $BUCKET/deep/
$E/target/release/reorder spacev spacev.1B.100.i8.bin spacev.1B.k128.bin spacev.1B.k128.perm spacev.1B.k128.centroids > spacev.1B.k128.sizes.csv
head -c $((201000000 * 100)) spacev.1B.k128.bin > spacev.201M.k128.raw
$E/target/release/crop spacev.1B.k128.bin spacev.100M.k128.bin 100000000 100 1
cp spacev.29316.100.i8.query spacev.query
gcloud storage cp spacev.201M.k128.raw spacev.100M.k128.bin spacev.query $BUCKET/spacevk/
$E/target/release/reorder msturing msturing.1B.100.f32.bin msturing.1B.k128.bin msturing.1B.k128.perm msturing.1B.k128.centroids > msturing.1B.k128.sizes.csv
head -c $((201000000 * 400)) msturing.1B.k128.bin > msturing.201M.k128.raw
$E/target/release/crop msturing.1B.k128.bin msturing.100M.k128.bin 100000000 100 4
curl -sS --fail -r 8- -o msturing.100K.100.f32.query https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/query100K.fbin
head -c 4000000 msturing.100K.100.f32.query > msturing.query
gcloud storage cp msturing.201M.k128.raw msturing.100M.k128.bin msturing.query $BUCKET/msturingk/
cd $E
for ds in deep spacevk msturingk; do
  sed -e "s|BUCKETURI|$BUCKET|g" -e "s/COMMIT/$COMMIT/g" -e "s/NAME/prep-$ds/g" -e "s/DS/$ds/g" -e "s/STEPS/both/g" 0015/gcpprep.json > job.json
  gcloud batch jobs submit prep-$ds --location=us-central1 --config=job.json
done
for ds in spacev msturing; do
  sed -e "s|BUCKETURI|$BUCKET|g" -e "s/COMMIT/$COMMIT/g" -e "s/NAME/prep-${ds}s/g" -e "s/DS/$ds/g" 0027/gcpprep.json > job.json
  gcloud batch jobs submit prep-${ds}s --location=us-central1 --config=job.json
done
for ds in sift deep spacevs msturings; do
  sed -e "s|BUCKETURI|$BUCKET|g" -e "s/COMMIT/$COMMIT/g" -e "s/NAME/ours-$ds-grow/g" -e "s/ARM/ours/g" -e "s/DATA/$ds/g" -e "s/MODE/grow/g" 0015/gcpfull.json > job.json
  gcloud batch jobs submit ours-$ds-grow --location=us-central1 --config=job.json
done
for ds in sift deep spacevs msturings; do
  sed -e "s|BUCKETURI|$BUCKET|g" -e "s/COMMIT/$COMMIT/g" -e "s/NAME/ours-$ds-churn/g" -e "s/ARM/ours/g" -e "s/DATA/$ds/g" -e "s/MODE/churn/g" 0015/gcpfull.json > job.json
  gcloud batch jobs submit ours-$ds-churn --location=us-central1 --config=job.json
done
for ds in sift deep spacevs msturings; do
  sed -e "s|BUCKETURI|$BUCKET|g" -e "s/COMMIT/$COMMIT/g" -e "s/NAME/odin-$ds-churn/g" -e "s/ARM/odin/g" -e "s/DATA/$ds/g" -e "s/MODE/churn/g" 0015/gcpfull.json > job.json
  gcloud batch jobs submit odin-$ds-churn --location=us-central1 --config=job.json
done
for ds in sift deep spacevs msturings; do
  sed -e "s|BUCKETURI|$BUCKET|g" -e "s/COMMIT/$COMMIT/g" -e "s/NAME/conda-$ds-churn/g" -e "s/ARM/conda/g" -e "s/DATA/$ds/g" -e "s/MODE/churn/g" 0015/gcpfull.json > job.json
  gcloud batch jobs submit conda-$ds-churn --location=us-central1 --config=job.json
done
sleep 64800
for ds in sift deep spacevs msturings; do for arm in odin conda; do
  gcloud batch jobs delete $arm-$ds-churn --location=us-central1 --quiet
done; done
mkdir -p runs
cd runs
declare -A GT=([sift]=sift/sift.gt.arrows [deep]=deep/deep.gt.arrows [spacevs]=spacevs/spacev.201M.s.gt.arrows [msturings]=msturings/msturing.201M.s.gt.arrows)
for ds in sift deep spacevs msturings; do
  gcloud storage cp $BUCKET/${GT[$ds]} gt_$ds.arrows
  gcloud storage cp $BUCKET/runs/ours-$ds-churn/ourschurn_$ds.arrows .
  uv run $E/0015/analyze.py ourschurn_$ds.arrows gt_$ds.arrows 100 recall_ourschurn_$ds.csv tp_ourschurn_$ds.csv
  gcloud storage cp $BUCKET/runs/odin-$ds-churn/odinchurn_$ds.arrows $BUCKET/runs/odin-$ds-churn/odinchurn_$ds.log .
  uv run $E/0015/analyze.py odinchurn_$ds.arrows gt_$ds.arrows 100 recall_odinchurn_$ds.csv tp_odinchurn_$ds.csv
  gcloud storage cp $BUCKET/runs/conda-$ds-churn/condachurn_$ds.arrows $BUCKET/runs/conda-$ds-churn/condachurn_$ds.mon .
  uv run $E/0015/analyze.py condachurn_$ds.arrows gt_$ds.arrows 100 recall_condachurn_$ds.csv tp_condachurn_$ds.csv
  gcloud storage cp $BUCKET/runs/ours-$ds-grow/oursfull_$ds.arrows .
  uv run $E/0015/analyze.py oursfull_$ds.arrows gt_$ds.arrows 100 recall_oursfull_$ds.csv tp_oursfull_$ds.csv
  uv run --with numpy --with pyarrow python $E/0015/intensity.py condachurn_$ds.arrows condachurn_$ds.mon intensity_condachurn_$ds.csv
done
cd $E
uv run tables.py runs .
uv run plot.py
