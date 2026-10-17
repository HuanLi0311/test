#!/bin/bash
# Launch 8 feature-extraction shards, one per GPU.
ROOT=/home/JJ_Group/lih2511/test/rna
PY=$ROOT/venv/bin/python
N=8

for split in train test; do
  for i in $(seq 0 $((N-1))); do
    nohup "$PY" "$ROOT/scripts/extract_shard.py" \
      --split "$split" --shard "$i" --nshards "$N" --max-len 1024 \
      > "$ROOT/logs/extract_${split}_$i.log" 2>&1 &
    echo "launched $split shard $i pid=$!"
  done
done
echo "all launched"
