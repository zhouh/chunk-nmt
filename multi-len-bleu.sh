#!/bin/bash

datadir=/home/zhouh/Data/nmt/

for i in $(seq 2 1 6)
do
echo '====== 0'$i'=======' 
python ./BLEUbyLength.py ../BLEU/multi-bleu.perl  $datadir/devntest/MT0${i}/MT0${i}.src ./test.result.chunk.$i $datadir/devntest/MT0${i}/reference
done
