:
INTERMEDIATE_LOAD=--intermediate_load

SAMPLE="ALL"

MOMENTS="\
denoise_after_cropping \
denoise_after_metadata \
denoise_after_preanalysis \
"

MOMENTS="denoise_after_cropping denoise_after_metadata"

MOMENTS="denoise_after_cropping"

MODES="\
denoise \
latent \
chiara_vae \
"

NOREDIRECT="--no_redirect_output"
SCVI_OPTS="--use_chiara_scvi"

NOGPU=
#NOGPU="--nogpu"

if [ $# -gt 0 ]
then
	SAMPLE="$1"
	shift
fi

export PYTHONUNBUFFERED=1

set -e



echo "RUNNING all MAIN scvi plots"
#for mode in denoise latent
for mode in full
do
	for moment in $MOMENTS
	do
		sh run_dca_as_module.sh $NOREDIRECT $SAMPLE $SCVI_OPTS   --denoise_moment=$moment $INTERMEDIATE_LOAD --thrashgpus --denoise_mode=$mode --denoiser=scvi --clustering_method=full $*
	done
done

echo "RUNNING all MAIN dca plots"
for mode in full
do
	for moment in $MOMENTS
	do
		sh run_dca_as_module.sh $NOREDIRECT $SAMPLE  --denoise_moment=$moment $INTERMEDIATE_LOAD $NOGPU --denoise_mode=$mode --use_chiara_dca --clustering_method=full $*
	done
done


echo "RUNNING all UNDENOISED plots"
sh run_dca_as_module.sh $NOREDIRECT $SAMPLE  --nodenoise $INTERMEDIATE_LOAD $NOGPU $*


exit 0

echo "RUNNING all EVALUATE plots for dca"
for mode in $MODES
do
	for moment in $MOMENTS
	do
		sh run_dca_as_module.sh $NOREDIRECT $SAMPLE  --dca_evaluate=hidden_size --noplot --denoise_moment=$moment --denoise_mode=$mode $INTERMEDIATE_LOAD $NOGPU $*
	done

	sh run_dca_as_module.sh $NOREDIRECT $SAMPLE  --dca_evaluate=hidden_size --noplot --denoise_moment=denoise_after_preanalysis --noregress $INTERMEDIATE_LOAD $NOGPU --denoise_mode=$mode $*
done
