#!/usr/bin/env bash
set -euo pipefail

# bytes [skip, end) of $url appended into $out, resumable; every dataset file
# starts with an 8-byte header, so skip=8 leaves only packed points on disk.
# end is the file's end, or the optional $stop when it falls short of that.
# curl, not wget: wget treats the 206 that any ranged request earns as an
# unexpected response and retries forever. Each pass asks for the bytes still
# missing and appends them, so a hiccup costs the current chunk rather than
# the whole file, and the size check is what calls it whole.
#
# --retry 0 is load-bearing: curl's own retry restarts the whole range and
# this loop appends, so an internal retry would write a prefix twice and
# overshoot. Resuming is the outer loop's job, from the size on disk.
rangefetch() {
  local url="$1" out="$2" skip="$3" stop="${4:-}"
  local total want got new stall=0
  total=$(curl -fsSIL "$url" | tr -d '\r' | awk 'tolower($1) == "content-length:" { n = $2 } END { print n }')
  [ -n "$total" ] || { echo "rangefetch: no Content-Length for $url" >&2; return 1; }
  [ -z "$stop" ] || [ "$stop" -ge "$total" ] || total=$stop
  want=$(( total - skip ))
  [ -f "$out" ] || : > "$out"
  got=$(stat -c %s "$out")
  while [ "$got" -lt "$want" ]; do
    curl -fsSL --retry 0 --speed-time 120 --speed-limit 4096 \
      -r "$(( skip + got ))-$(( total - 1 ))" "$url" >> "$out" || true
    new=$(stat -c %s "$out")
    if [ "$new" -gt "$want" ]; then
      echo "rangefetch: $out overshot to $new bytes, restarting" >&2
      : > "$out"
      new=0
    fi
    if [ "$new" -le "$got" ]; then
      stall=$(( stall + 1 ))
      [ "$stall" -lt 10 ] || { echo "rangefetch: $out no progress at $got bytes" >&2; return 1; }
      sleep 10
    else
      stall=0
      echo "rangefetch: $out $new / $want bytes"
    fi
    got=$new
  done
  [ "$got" -eq "$want" ] || { echo "rangefetch: $out is $got bytes, want $want" >&2; return 1; }
}

# end byte of the first $3 rows, $2 bytes each, past a $1-byte header; prints
# nothing when $3 is empty. $3 is digits with an optional K, M or B suffix.
limit() {
  [ -n "${3:-}" ] || return 0
  local rows="${3%[KMB]}"
  case "$3" in
    *K) rows=$(( rows * 1000 )) ;;
    *M) rows=$(( rows * 1000000 )) ;;
    *B) rows=$(( rows * 1000000000 )) ;;
  esac
  echo $(( $1 + rows * $2 ))
}

# 1billion 128d u8
sift() {
  rangefetch https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/base.1B.u8bin "sift.${1:-1B}.128.u8.bin" 8 "$(limit 8 128 "${1:-}")"
}

# query vectors for sift
sift_q() {
  rangefetch https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/bigann/query.public.10K.u8bin "sift.${1:-10K}.128.u8.query" 8 "$(limit 8 128 "${1:-}")"
}

# 1billion 96d f32
deep() {
  rangefetch https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/base.1B.fbin "deep.${1:-1B}.96.f32.bin" 8 "$(limit 8 384 "${1:-}")"
}

# query vectors for deep
deep_q() {
  rangefetch https://storage.yandexcloud.net/yandex-research/ann-datasets/DEEP/query.public.10K.fbin "deep.${1:-10K}.96.f32.query" 8 "$(limit 8 384 "${1:-}")"
}

# 1billion 100d f32
msturing() {
  rangefetch https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/base1b.fbin "msturing.${1:-1B}.100.f32.bin" 8 "$(limit 8 400 "${1:-}")"
}

# query vectors for msturing
msturing_q() {
  rangefetch https://comp21storage.z5.web.core.windows.net/comp21/MSFT-TURING-ANNS/testQuery10K.fbin "msturing.${1:-10K}.100.f32.query" 8 "$(limit 8 400 "${1:-}")"
}

# 1billion 200d f32
t2i() {
  rangefetch https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/base.1B.fbin "t2i.${1:-1B}.200.f32.bin" 8 "$(limit 8 800 "${1:-}")"
}

# query vectors for t2i
t2i_q() {
  rangefetch https://storage.yandexcloud.net/yandex-research/ann-datasets/T2I/query.public.100K.fbin "t2i.${1:-100K}.200.f32.query" 8 "$(limit 8 800 "${1:-}")"
}

# 1402020720 100d i8
# NOTE: these datasets are stored via Git LFS -- github.com/.../raw/...
# only serves the ~130-byte LFS pointer, not the real ~4GiB shard, so this
# must go through media.githubusercontent.com's LFS media endpoint instead.
# Only the first shard carries the 8-byte header. With a row count, shards
# are fetched until that many rows are in, the last one cut short.
spacev() {
  local url="https://media.githubusercontent.com/media/microsoft/SPTAG/main/datasets/SPACEV1B/vectors.bin"
  local i skip part left
  left=$(limit 0 100 "${1:-}")
  for i in $(seq 1 33); do
    part="spacev.${1:-1B}.100.i8.bin.part${i}"
    skip=0
    [ "$i" -ne 1 ] || skip=8
    rangefetch "${url}/vectors_${i}.bin" "$part" "$skip" "${left:+$(( skip + left ))}"
    cat "$part" >> "spacev.${1:-1B}.100.i8.bin"
    [ -z "$left" ] || left=$(( left - $(stat -c %s "$part") ))
    rm "$part"
    [ -z "$left" ] || [ "$left" -gt 0 ] || break
  done
}

spacev_q() {
  rangefetch https://media.githubusercontent.com/media/microsoft/SPTAG/main/datasets/SPACEV1B/query.bin "spacev.${1:-29316}.100.i8.query" 8 "$(limit 8 100 "${1:-}")"
}

# each argument names a dataset above, or name[COUNT] for its first COUNT
# rows, COUNT being digits with an optional K, M or B suffix
for arg in "$@"; do
  if [[ "$arg" =~ ^([a-z0-9_]+)\[([0-9]+[KMB]?)\]$ ]]; then
    "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
  else
    $arg
  fi
done
