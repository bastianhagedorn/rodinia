#!/bin/bash

if [ "$#" -ne 5 ]; then
				echo "Illegal number of parameters (enter platform, device, iterations, rows and columns)"
				exit -1
fi

N=$3

RAWDATA="srad.raw"
OUT1="srad1.out"
OUT2="srad2.out"

if [ -e $RAWDATA ]
then
    rm $RAWDATA
fi

if [ -e $OUT1 ]
then
    rm $OUT1
fi

if [ -e $OUT2 ]
then
    rm $OUT2
fi

for i in $(seq 1 $N)
do
./srad 1 0.5 $4 $5 $1 $2 >> srad.raw
done

cat srad.raw | grep SRAD | awk '{print $2}' > srad1.out
cat srad.raw | grep SRAD | awk '{print $4}' > srad2.out
cat srad1.out
cat srad2.out
