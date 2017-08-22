#!/bin/bash
DATADIR=/home/v1bhaged/benchmarks/rodinia/data

if [ "$#" -ne 4 ]; then
				echo "Illegal number of parameters (enter platform, device, input size and layers)"
				exit -1
fi


# prints kernel runtime in nanoseconds to hotspot.out using kernel event times
rm hotspot3D.raw
rm hotspot3D.out

echo ./3D $3 $4 1 $DATADIR/hotspot3D/power_$3x$4 $DATADIR/hotspot3D/temp_$3x$4  output.out 


for i in {1..10}
do
./3D $3 $4 1 $DATADIR/hotspot3D/power_$3x$4 $DATADIR/hotspot3D/temp_$3x$4  output.out >> hotspot3D.raw
done

cat hotspot3D.raw | grep DEBUG | awk '{print $4}' > hotspot3D.out
cat hotspot3D.out
