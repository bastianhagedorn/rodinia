
#!/bin/bash
if [ "$#" -ne 1 ]; then
				echo "Illegal number of parameters (enter input size)"
				exit -1
fi

# prints kernel runtime in nanoseconds to hotspot.out using kernel event times
rm hotspot.raw
rm hotspot.out

echo temp_$1

for i in {1..10}
do
./hotspot $1 1 1 ../../data/hotspot/temp_$1 ../../data/hotspot/power_$1 output.out >> hotspot.raw
done

cat hotspot.raw | grep DEBUG | awk '{print $4}' > hotspot.out
cat hotspot.out
