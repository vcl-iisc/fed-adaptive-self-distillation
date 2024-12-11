datasetname="CIFAR100"
num_clients="500"
partition="Dirichlet"
dir_val="0.300"
reg_val="20"
seed_val="2024"
m_name="cifar100"
us="_"
hs="/"
#unbalanced_str="1.000000"

f_name1="Output"
#f_name2=$datasetname$us$num_clients$us$partition$us$dir_val$us$unbalanced_str
f_name2=$datasetname$us$num_clients$us$partition$us$dir_val

feddyn_noreg_str="FedDyn_noreg"
feddyn_reg_str="FedDyn_reg"

feddyn_noreg_str1="FedDyn_noreg1"
feddyn_reg_str1="FedDyn_reg1"

feddyn_noreg_str2="FedDyn_noreg2"
feddyn_reg_str2="FedDyn_reg2"

feddyn_noreg_str3="FedDyn_noreg3"
feddyn_reg_str3="FedDyn_reg3"


fedavg_noreg_str="FedAvg_noreg"
fedavg_noreg_str1="FedAvg_noreg1"
fedavg_noreg_str2="FedAvg_noreg2"
fedavg_noreg_str3="FedAvg_noreg3"
fedavgreg_reg_str="FedAvgReg_reg"
fedntd_noreg="FedNTD_noreg"
fedntd_reg="FedNTD_reg"
fedavgstr="FedAvg"
feddynstr="FedDyn"
fedproxstr="FedProx"
fedspeedstr="FedSpeed"
fedspeedregstr="FedSpeed_reg"
fedspeednoregstr="FedSpeed_noreg"

fedprox_noreg_str="FedProx_noreg"
fedprox_reg_str="FedProx_reg"

fedprox_noreg_str1="FedProx_noreg1"
fedprox_reg_str1="FedProx_reg1"

fedprox_noreg_str2="FedProx_noreg2"
fedprox_reg_str2="FedProx_reg2"

fedprox_noreg_str3="FedProx_noreg3"
fedprox_reg_str3="FedProx_reg3"

fedntdstr="FedAvgReg"
fedavgregstr="FedAvgReg"
feddisco_noreg="FedDisco_noreg"
feddisco_reg="FedDisco_reg"

fedavgreg_uniform="fedavgreg_uniform"
fedavgreg_onlykl="fedavgreg_onlykl"
fedavgreg_onlybeta='fedavgreg_only_beta' 
fedavgreg_hl="fedavgreg_hl"
fedavgreg_beta1="fedavgreg_beta1"
fedavgreg_beta2="fedavgreg_beta2"
fedavgreg_beta3="fedavgreg_beta3"
fedavgreg_beta4="fedavgreg_beta4"
fedavgreg_beta5="fedavgreg_beta5"
fedavgreg_beta6="fedavgreg_beta6"
fedavgreg_unidist="fedavgreg_unidist"

################# fedavg ##########
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=0 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvg --lamda=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --seed=$seed_val --ntd=0 --uniform_distill=1 --entropy_flag=1 --dist_beta_kl=1.0 --dist_beta=1
##
#old_f=$f_name1$hs$f_name2$hs$fedavgstr
##
###rm -rf ${old_f}/*.pt
#rm -rf ${old_f}/100_com_*.pt
#rm -rf ${old_f}/200_com_*.pt
#rm -rf ${old_f}/300_com_*.pt
#rm -rf ${old_f}/400_com_*.pt
#
#rm -rf ${old_f}/*param*
##
#new_f=$f_name1$hs$f_name2$hs$fedavg_noreg_str1
##
#mv $old_f $new_f
#
################ fedavg + reg ############
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvgReg --lamda=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --seed=$seed_val --ntd=0 --uniform_distill=0 --entropy_flag=1 --dist_beta_kl=1.0 --dist_beta=1
##
#old_f=$f_name1$hs$f_name2$hs$fedavgregstr
##
#rm -rf ${old_f}/*.pt
##
#rm -rf ${old_f}/*param*
##
##new_f=$f_name1$hs$f_name2$hs$fedavgreg_reg_str
#new_f=$f_name1$hs$f_name2$hs$fedavgreg_beta1
##
#mv $old_f $new_f






#seed_val="2025"

#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvgReg --lamda=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --seed=$seed_val --ntd=0 --uniform_distill=0 --entropy_flag=1 --dist_beta_kl=1.0 --dist_beta=1
##
#old_f=$f_name1$hs$f_name2$hs$fedavgregstr
##
#rm -rf ${old_f}/*.pt
##
#rm -rf ${old_f}/*param*
##
##new_f=$f_name1$hs$f_name2$hs$fedavgreg_reg_str
#new_f=$f_name1$hs$f_name2$hs$fedavgreg_beta2
##
#mv $old_f $new_f
#
#
#seed_val="2026"
#
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvgReg --lamda=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --seed=$seed_val --ntd=0 --uniform_distill=0 --entropy_flag=1 --dist_beta_kl=1.0 --dist_beta=1
##
#old_f=$f_name1$hs$f_name2$hs$fedavgregstr
##
#rm -rf ${old_f}/*.pt
##
#rm -rf ${old_f}/*param*
##
##new_f=$f_name1$hs$f_name2$hs$fedavgreg_reg_str
#new_f=$f_name1$hs$f_name2$hs$fedavgreg_beta3
##
#mv $old_f $new_f


#seed_val="2025"
#
seed_val="2029"
############### fedntd no reg ###############
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvgReg --lamda=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --seed=$seed_val --ntd=1 --uniform_distill=1 --entropy_flag=0 --dist_beta_kl=0.0 --dist_beta=0
#
#old_f=$f_name1$hs$f_name2$hs$fedavgregstr
#
##rm -rf ${old_f}/*.pt
#
#rm -rf ${old_f}/100_com_*.pt
#rm -rf ${old_f}/200_com_*.pt
#rm -rf ${old_f}/300_com_*.pt
#rm -rf ${old_f}/400_com_*.pt

#rm -rf ${old_f}/*param*
#
#new_f=$f_name1$hs$f_name2$hs$fedntd_noreg
#
#mv $old_f $new_f


#seed_val="2026"
################ fedntd  + reg ################
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvgReg --lamda=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --seed=$seed_val --ntd=1 --uniform_distill=0 --entropy_flag=1 --dist_beta_kl=1.0 --dist_beta=1
##
#old_f=$f_name1$hs$f_name2$hs$fedavgregstr
##
#rm -rf ${old_f}/*.pt
##
#rm -rf ${old_f}/*param*
##
#new_f=$f_name1$hs$f_name2$hs$fedntd_reg
##
#mv $old_f $new_f
#
#
############### feddyn #####################
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=0 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedDyn --lamda=0.0 --mu_var=0.0 --epoch=10 --lr_decay_per_round=0.998 --learning_rate=0.1 --seed=$seed_val --ntd=0 --uniform_distill=0 --entropy_flag=1 --dist_beta_kl=0 --dist_beta=0
###
#old_f=$f_name1$hs$f_name2$hs$feddynstr
###
#rm -rf ${old_f}/*.pt
###
#rm -rf ${old_f}/*param*
###
#new_f=$f_name1$hs$f_name2$hs$feddyn_noreg_str1
###
#mv $old_f $new_f
#
#

############# feddyn + reg #################
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedDyn --lamda=$reg_val --mu_var=0.0 --epoch=10 --lr_decay_per_round=0.998 --learning_rate=0.1 --seed=$seed_val --ntd=0 --uniform_distill=0 --entropy_flag=1 --dist_beta_kl=1.0 --dist_beta=1.0
####
#old_f=$f_name1$hs$f_name2$hs$feddynstr
####
#rm -rf ${old_f}/*.pt
####
#rm -rf ${old_f}/*param*
####
#new_f=$f_name1$hs$f_name2$hs$feddyn_reg_str1
#mv $old_f $new_f

seed_val="2027"
############## fedprox_noreg #################
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=0 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedProx --lamda=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --seed=$seed_val --ntd=0 --uniform_distill=0 --entropy_flag=1 --dist_beta_kl=1.0 --dist_beta=1
##
#old_f=$f_name1$hs$f_name2$hs$fedproxstr
##
#rm -rf ${old_f}/*.pt
###
##rm -rf ${old_f}/100_com_*.pt
##rm -rf ${old_f}/200_com_*.pt
##rm -rf ${old_f}/300_com_*.pt
##rm -rf ${old_f}/400_com_*.pt
##
#rm -rf ${old_f}/*param*
###
#new_f=$f_name1$hs$f_name2$hs$fedprox_noreg_str2
###
#mv $old_f $new_f

############## fedprox_+ reg #################
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedProx --lamda=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --seed=$seed_val --ntd=0 --uniform_distill=0 --entropy_flag=1 --dist_beta_kl=1.0 --dist_beta=1
###
#old_f=$f_name1$hs$f_name2$hs$fedproxstr
###
#rm -rf ${old_f}/*.pt
###
#rm -rf ${old_f}/*param*
###
#new_f=$f_name1$hs$f_name2$hs$fedprox_reg_str2
###
#mv $old_f $new_f



############ fedspeed_noreg #################
python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=0 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedSpeed --lamda=0.0 --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --seed=$seed_val --ntd=0 --uniform_distill=0 --entropy_flag=1 --dist_beta_kl=1.0 --dist_beta=1
#
old_f=$f_name1$hs$f_name2$hs$fedspeedstr
#
rm -rf ${old_f}/*.pt
#
#rm -rf ${old_f}/100_com_*.pt
#rm -rf ${old_f}/200_com_*.pt
#rm -rf ${old_f}/300_com_*.pt
#rm -rf ${old_f}/400_com_*.pt

rm -rf ${old_f}/*param*
#
new_f=$f_name1$hs$f_name2$hs$fedspeednoregstr
#
mv $old_f $new_f



############# fedspeed_+ reg #################
python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedSpeed --lamda=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --seed=$seed_val --ntd=0 --uniform_distill=0 --entropy_flag=1 --dist_beta_kl=1.0 --dist_beta=1
#
old_f=$f_name1$hs$f_name2$hs$fedspeedstr
#
rm -rf ${old_f}/*.pt
#
rm -rf ${old_f}/*param*
#
new_f=$f_name1$hs$f_name2$hs$fedspeedregstr
#
mv $old_f $new_f
#fedspeedstr="FedSpeed"

############### fedavg + disco ##########
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=0 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvg --lamda=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --seed=$seed_val --ntd=0 --uniform_distill=0 --entropy_flag=1 --dist_beta_kl=1.0 --dist_beta=1 --disco=1
##
#old_f=$f_name1$hs$f_name2$hs$fedavgstr
##
###rm -rf ${old_f}/*.pt
##
#rm -rf ${old_f}/100_com_*.pt
#rm -rf ${old_f}/200_com_*.pt
#rm -rf ${old_f}/300_com_*.pt
#rm -rf ${old_f}/400_com_*.pt
#
#rm -rf ${old_f}/*param*
##
#new_f=$f_name1$hs$f_name2$hs$feddisco_noreg
##
#mv $old_f $new_f


############### fedavg + reg + disco ############
#python example_cifar_10.py --model_name=$m_name --dataset_name=$datasetname --add_reg=1 --unbalanced_sgm=0 --rule=$partition --rule_arg=$dir_val  --alg_name=FedAvgReg --lamda=$reg_val --mu_var=0.0 --epoch=5 --lr_decay_per_round=0.998 --learning_rate=0.1 --seed=$seed_val --ntd=0 --uniform_distill=0 --entropy_flag=1 --dist_beta_kl=1.0 --dist_beta=1 --disco=1
##
#old_f=$f_name1$hs$f_name2$hs$fedavgregstr
##
###rm -rf ${old_f}/*.pt
##
#rm -rf ${old_f}/*param*
##
#new_f=$f_name1$hs$f_name2$hs$feddisco_reg
##
#mv $old_f $new_f



