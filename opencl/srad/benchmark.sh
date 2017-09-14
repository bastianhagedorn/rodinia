#!/bin/bash
N=100

if [ "$#" -ne 5 ]; then
				echo "Illegal number of parameters (enter platform, device, iterations, rows and columns)"
				exit -1
fi


# prints kernel runtime in nanoseconds to srad.out using kernel event times

rm srad1.out
rm srad2.out
rm srad.raw

for i in $(seq 1 $N)
do
./srad $3 0.5 $4 $5 $1 $2 >> srad.raw
done

cat srad.raw | grep SRAD | awk '{print $2}' > srad1.out
cat srad.raw | grep SRAD | awk '{print $4}' > srad2.out
cat srad1.out
cat srad2.out
