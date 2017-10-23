#!/bin/bash
DATADIR=/home/v1bhaged/benchmarks/rodinia/data

if [ "$#" -ne 4 ]; then
				echo "Illegal number of parameters (enter platform, device, input size and number of iterations)"
				exit -1
fi

N=$4

RAWDATA="hotspot.raw"
OUTDATA="hotspot.out"
if [ -e $RAWDATA ]
then
    rm $RAWDATA 
fi
if [ -e $OUTDATA ]
then
    rm $OUTDATA 
fi

for i in $(seq 1 $N)
do
./hotspot $3 1 1 $DATADIR/hotspot/temp_$3 $DATADIR/hotspot/power_$3 output.out $1 $2 >> hotspot.raw
done

cat hotspot.raw | grep DEBUG | awk '{print $4}' > hotspot.out
cat hotspot.out
