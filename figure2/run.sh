#!/bin/bash
set -e
cd "$(dirname "$0")"
: "${DATA:?}" "${WORK:?}"
export DATA WORK
git submodule update --init --recursive .
export E=$PWD
mkdir -p $DATA $WORK
cd $DATA
bash $E/download.sh 'sift[20M]' 'deep[20M]' 'spacev[20M]' 'msturing[20M]' 't2i[20M]'
mv sift.20M.128.u8.bin sift.20M.128.bin
mv deep.20M.96.f32.bin deep.20M.96.bin
mv spacev.20M.100.i8.bin spacev.20M.100.bin
mv t2i.20M.200.f32.bin t2i.20M.200.bin
cd -

bash $E/0017/run.sh compile
uv run $E/0007/shuffle.py $DATA/spacev.20M.100.bin $DATA/spacev.20M.100.shuf.bin 100
for ds in sift deep spacev msturing t2i; do
    bash $E/0017/run.sh cut $ds
    bash $E/0017/run.sh build $ds
    bash $E/0017/run.sh hop $ds 64
done

python3 $E/motivation_tables.py figure2 $WORK/logs > candidates_per_insert.csv
uv run plot.py
