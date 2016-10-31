#!/usr/bin/python
import sys,os
cmd='python /prj/neo-nas/users/stalathi/sachin-repo/Neo/SysPTSD/Lasagne/Learn_Examples/cifar10_binary.py -C -A -c \
--epochs 100 --learning-rate 0.01 --data-dir /prj/neo_lv/user/stalathi/DataSets/cifar-10-batches-py \
--memo Cifar10_Ternary_Augment_Cool_0.01 --train --quantization ternary'

os.system(cmd)
