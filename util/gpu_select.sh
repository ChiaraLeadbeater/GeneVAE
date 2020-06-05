:

VALS=`gpustat --no-color | cut -c 31-32`


cnt=0
idx=0
MIN=100
for i in $VALS
do
	cnt=$((cnt+1))
	if [ $cnt -eq 1 ]
	then
		continue
	else
		n=$((i))
		if [ $n -lt $MIN ]
		then
			MIN=$((n))
			GPU=$idx
		fi
		idx=$((idx+1))
	fi
done

if [ $MIN -lt 90 ]
then
	echo $GPU
else
	echo "-1"
fi
