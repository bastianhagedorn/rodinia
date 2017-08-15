
#!/bin/bash
if [ "$#" -ne 3 ]; then
				echo "Illegal number of parameters (enter platform, device and input size)"
				exit -1
fi

# prints kernel runtime in nanoseconds to hotspot.out using kernel event times
rm hotspot.raw
rm hotspot.out

for i in {1..10}
do
./hotspot $3 1 1 ../../data/hotspot/temp_$3 ../../data/hotspot/power_$3 output.out $1 $2 >> hotspot.raw
done

cat hotspot.raw | grep DEBUG | awk '{print $4}' > hotspot.out
cat hotspot.out
