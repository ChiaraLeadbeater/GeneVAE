:

FOETAL=foetal.smartseq2.py
STABLE=no
REDIRECT=yes
REDIRECT_OUTPUT="out.development"

echo $* >> dca_run.log

while [ $# -gt 0 ]; do
	if [ $1 = '--use_stable_version' ]
	then
		FOETAL="stable.$FOETAL"
		STABLE=yes
		REDIRECT_OUTPUT="out.stable"
		shift
	elif [ $1 = '--no_redirect_output' ]
	then
		REDIRECT=no
		shift
	elif [ $1 = '--redirect_unique' ]
	then
		DT=`date +%Y%m%d_%H%M%S`
		REDIRECT_OUTPUT="out.$DT"
		shift
	else
		#case "$1" in 
			#--*)
			#echo "Bad option $1"
			#exit 1
		#esac
		break
	fi
done


echo "$REDIRECT_OUTPUT" >> dca_run.log

#if [ $1 = '--use_stable_version' ]
#then
#	FOETAL="stable.$FOETAL"
#	shift
#fi

echo "Using $FOETAL"

if [ "$REDIRECT" = 'yes' ]
then
	export PYTHONUNBUFFERED=1
fi


if [ $# -lt 1 ]
then
	echo "Need a number or ALL"
	exit
fi

if [ $1 = "ALL" ]
then
	LIST="1 2 3 4 5"
else
	LIST=$1
fi
shift


if [ -z "$FOETAL_PYTHON_PATH" ]
then
	FOETAL_PYTHON_PATH="/home/cnl29/Project/src"
fi
export PYTHONPATH="$FOETAL_PYTHON_PATH:$PYTHONPATH"

BASE_DIR=
if [ -n "$FOETAL_SCRATCH_PATH" ]
then
	BASE_DIR="--base_dir=$FOETAL_SCRATCH_PATH"
fi


echo $PYTHONPATH
echo "$FOETAl_SCRATCH_PATH"
echo "$BASE_DIR"


set -e


if [ -z "$PYTHON_VERSION" ]
then
	PYTHON_VERSION=python3.6
fi

echo "Using $PYTHON_VERSION"
echo "Initial CUDA_VISIBLE_DEVICES setting = $CUDA_VISIBLE_DEVICES"
if [ -n "$PYTHONUNBUFFERED" ]
then
	echo "PYTHONUNBUFFERED is set"
fi

for i in $LIST
do
	if [ "$REDIRECT" = "yes" ]
	then
		stdbuf -o0 $PYTHON_VERSION $FOETAL --sample=$i $BASE_DIR $* > $REDIRECT_OUTPUT 2>&1
	else
		stdbuf -o0 $PYTHON_VERSION $FOETAL --sample=$i $BASE_DIR $*
	fi
done

echo "SUCCESS" >> dca_run.log

exit 0
