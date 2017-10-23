#!/bin/bash
DATADIR=/home/v1bhaged/benchmarks/rodinia/data

if [ "$#" -ne 5 ]; then
				echo "Illegal number of parameters (enter platform, device, input size, layers and iterations)"
				exit -1
fi

N=$5

RAWDATA="hotspot3D.raw"
OUTDATA="hotspot3D.out"

if [ -e $RAWDATA ]
then
    rm hotspot3D.raw
fi
if [ -e $OUTDATA ]
then
    rm hotspot3D.out
fi

echo ./3D $3 $4 $5 $DATADIR/hotspot3D/power_$3x$4 $DATADIR/hotspot3D/temp_$3x$4  output.out $1 $2 

for i in $(seq 1 $N)
do
    ./3D $3 $4 1 $DATADIR/hotspot3D/power_$3x$4 $DATADIR/hotspot3D/temp_$3x$4  output.out 0 0 >> hotspot3D.raw
done

cat hotspot3D.raw | grep DEBUG | awk '{print $3}' > hotspot3D.out
cat hotspot3D.out
