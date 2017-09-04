#!/bin/bash
DATADIR=/home/odroid/benchmarks/rodinia/data 
N=10

if [ "$#" -ne 5 ]; then
				echo "Illegal number of parameters (enter platform, device, input size, layers and iterations)"
				exit -1
fi


# prints kernel runtime in nanoseconds to hotspot.out using kernel event times
rm hotspot3D.raw
rm hotspot3D.out

echo ./3D $3 $4 $5 $DATADIR/hotspot3D/power_$3x$4 $DATADIR/hotspot3D/temp_$3x$4  output.out 


for i in $(seq 1 $N)
do
./3D $3 $4 $5 $DATADIR/hotspot3D/power_$3x$4 $DATADIR/hotspot3D/temp_$3x$4  output.out 0 0 >> hotspot3D.raw
done

cat hotspot3D.raw | grep Time: | awk '{print $2}' > hotspot3D.out
cat hotspot3D.out
