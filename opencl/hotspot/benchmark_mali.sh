#!/bin/bash

# prints kernel runtime in nanoseconds to hotspot.out using kernel event times

rm hotspot.raw
rm hotspot.out

for i in {1..10}
do
./hotspot 512 1 1 ../../data/hotspot/temp_512 ../../data/hotspot/power_512 output.out >> hotspot.raw
done

cat hotspot.raw | grep DEBUG | awk '{print $4}' > hotspot.out
cat hotspot.out
