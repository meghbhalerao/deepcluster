######START OF EMBEDDED SGE COMMANDS ##########################
#$ -S /bin/bash
#$ -cwd
#$ -N run_train
#$ -M 16ee234.megh@nitk.edu.in #### email to nofity with following options/scenarios
#$ -m a #### send mail in case the job is aborted
#$ -m b #### send mail when job begins
#$ -m e #### send mail when job ends
#$ -m s #### send mail when job is suspended
#$ -l h_vmem=40G
#$ -l gpu=1
############################## END OF DEFAULT EMBEDDED SGE COMMANDS #######################
CUDA_VISIBLE_DEVICES=`get_CUDA_VISIBLE_DEVICES` || exit
export CUDA_VISIBLE_DEVICES 
source activate dss

DIR="./data/clipart/"
ARCH="alexnet"
LR=0.05
WD=-5
K=126
WORKERS=4
EXP="./output/"
##PYTHON="/private/home/${USER}/test/conda/bin/python"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0 python main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}
