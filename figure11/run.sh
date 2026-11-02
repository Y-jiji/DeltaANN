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
cargo build --release --bin gtbrute
mkdir -p $DATA
cd $DATA
bash $E/download.sh sift sift_q
$E/target/release/gtbrute u8 128 sift.1B.128.u8.bin sift.10K.128.u8.query sift1b.gt.arrows sift1b.scratch
gcloud storage cp sift1b.gt.arrows $BUCKET/sift1b/
cd $E
sed -e "s|BUCKETURI|$BUCKET|g" -e "s/COMMIT/$COMMIT/g" -e "s/FNCT/578c7a3/g" -e "s/NAME/build-sift1b/g" -e "s/EXTRA//g" 0028/gcpbuild.json > job.json
gcloud batch jobs submit build-sift1b --location=us-central1 --config=job.json
for mode in grow churn; do
  sed -e "s|BUCKETURI|$BUCKET|g" -e "s/COMMIT/$COMMIT/g" -e "s/FNCT/578c7a3/g" -e "s/NAME/ours-$mode/g" -e "s/MODE/$mode/g" 0028/gcprun.json > job.json
  gcloud batch jobs submit ours-$mode --location=us-central1 --config=job.json
done
sed -e "s|BUCKETURI|$BUCKET|g" -e "s/COMMIT/$COMMIT/g" -e "s/PIPEANN/d4e7da2/g" -e "s/FNCT/578c7a3/g" -e "s/FRESH/2c185ab/g" -e "s/NAME/odin-grow/g" -e "s/MODE/grow/g" -e "s/CPUS/0/g" -e "s/KNOBS//g" -e "s/CPUMILLI/15000/g" -e "s/MEMORYMIB/100000/g" -e "s/RUNMAX/86400s/g" -e "s/MACHINE/z3-highmem-16-highlssd/g" 0028/gcpodin.json > job.json
gcloud batch jobs submit odin-grow --location=us-central1 --config=job.json
sed -e "s|BUCKETURI|$BUCKET|g" -e "s/COMMIT/$COMMIT/g" -e "s/PIPEANN/d4e7da2/g" -e "s/FNCT/578c7a3/g" -e "s/FRESH/2c185ab/g" -e "s/NAME/odin-churn/g" -e "s/MODE/churn/g" -e "s/CPUS/16/g" -e "s/KNOBS/budget=13500000/g" -e "s/CPUMILLI/21000/g" -e "s/MEMORYMIB/150000/g" -e "s/RUNMAX/86400s/g" -e "s/MACHINE/z3-highmem-22-highlssd/g" 0028/gcpodin.json > job.json
gcloud batch jobs submit odin-churn --location=us-central1 --config=job.json
sleep 46800
gcloud batch jobs delete odin-churn --location=us-central1 --quiet
mkdir -p runs
cd runs
gcloud storage cp $BUCKET/sift1b/sift1b.gt.arrows .
for job in ours-churn:ourschurn ours-grow:oursgrow odin-churn:odinchurn odin-grow:odingrow; do
  T=${job#*:}_sift1b
  gcloud storage cp $BUCKET/runs/${job%%:*}/$T.arrows .
  uv run $E/0028/analyze.py $T.arrows sift1b.gt.arrows 899 recall_$T.csv tp_$T.csv
done
cd $E
uv run tables.py runs .
uv run plot.py
