#!/bin/bash

#set -x

# python virtualenv setup
. /etc/bash_completion.d/virtualenvwrapper
WORKON_HOME=/prj/neo-nas/users/python_virtualenvs
workon theano_nokeras
export CUDA_LAUNCH_BLOCKING=1 ## required for theano profiling
export THEANO_FLAGS='floatX=float32, profile=False, device=gpu, base_compiledir=/tmp/stalathi/.theano'
export PATH=/prj/neo/pkg/cuda7/bin:$PATH
export CPATH=/prj/neoci/tools/prebuilt/cuDNN/7.0-v4.0/linux-x64/cuda/include
export LD_LIBRARY_PATH=/prj/neoci/tools/prebuilt/cuDNN/7.0-v4.0/linux-x64/cuda/lib64:/prj/neoci/releases/dl/dl-5.62.0.1377/x86_64-linux-clang/lib:/prj/neoci/releases/dl/dl-5.62.0.1377/x86_64-linux-clang/ThirdParty/cuda-convnet2/lib/python:/prj/neo/pkg/cuda7/lib64:/prj/neo/pkg/cuda7/lib:/prj/neo/pkg/cuda6/lib64:/prj/neo/pkg/cuda6/lib:/usr/lib64/nvidia/:/usr/local/cuda/lib
export LIBRARY_PATH=/prj/neoci/tools/prebuilt/cuDNN/7.0-v4.0/linux-x64/cuda/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/prj/neoci/releases/dl/dl-5.62.0.1377/x86_64-linux-clang/lib/python:/prj/neoci/releases/dl/dl-5.62.0.1377/x86_64-linux-clang/ThirdParty/caffe/python:/prj/neoci/releases/dl/dl-5.62.0.1377/x86_64-linux-clang/ThirdParty/cuda-convnet2/lib/python:/prj/neoci/releases/dl/dl-5.62.0.1377/x86_64-linux-clang/ThirdParty/cuda-convnet2/lib/python/python_util:/prj/neoci/releases/dl/dl-5.62.0.1377/x86_64-linux-clang/ThirdParty/lib/python:/prj/neo_lv/users/stalathi/SysR-repo/DeepLearning/RNN:/prj/neo-nas/users/stalathi/sachin-repo/Neo/SysPTSD/Lasagne
#export LD_LIBRARY_PATH=/prj/neo/pkg/cuda7/lib64:$LD_LIBRARY_PATH
#export PYTHONPATH=/prj/neo_scratch/users/stalathi/Lasagne:/prj/neo_scratch/users/stalathi/Git_Keras/keras/:/prj/neo-nas/users/stalathi/sachin-repo/Neo/SysRealization/DeepLearning/RNN/seya:$PYTHONPATH
#compiler setup
VER=4.6.4
GCCROOT=/prj/neo-nas/users/gnu/gcc-${VER}
export PATH=${GCCROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${GCCROOT}/lib64:${GCCROOT}/lib32:${GCCROOT}/lib:${LD_LIBRARY_PATH}

# configure theano
SCRIPTDIR=$(dirname $(readlink -f $0))
export THEANORC=$(readlink -f ${SCRIPTDIR}/theano.rc)

