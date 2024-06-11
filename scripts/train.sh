

# sleep 6.5 hours
# sleep 23400
# sleep 2400

extra_params=''

gpu=0
for dataset in  cifar10  # cifar10 # cifar100
do
    for method in ExtrapSAM # SAGM  # VASSO ASAM SAGM Lookahead ExtrapSAM_New SGD SGDNes SAM ExtrapSAM # VASSO ASAM SAGM Lookahead w  ASAM SAGM Lookahead # SGDNes VASSO ExtrapSAM  Lookahead # SGD SAM # SGDNes VASSO ExtrapSAM  Lookahead ASAM SAGM # SGD SAM SGDNes VASSO # SAGM Lookahead ASAM ExtrapSAM # SGDNes # VASSO SAGM Lookahead # ASAM ExtrapSAM # SGD SAM # RSAM  SGD SAGM 
    do 
        for model in vit # WRN-28-10 # resnet18  VGG19  WRN-28-10  # densenet121 # vit  # pyramidnet272 vit 
        do 
            for seed in  1 2 3
                do

                # if [ $model == "vit" ]; then
                #     use_adam=1
                # else
                #     lr=0.1
                # fi
                # lr=3e-4

                # PYTHONPATH=. screen -dm python methods/train_${method}.py --gpu ${gpu} --dataset ${dataset} --model resnet18 --train_type='LA_SAM' --queue_size 2  --seed ${seed} \
                # --save-path outputs/${dataset}/${method}_NEW/${seed}

                # for rho in 0.01 0.03 0.07 # 0.05 0.1
                # do

                if [ $use_adam == 1 ]; then
                    echo "Using Adam for ${method} on ${dataset} with ${model} and seed ${seed}"
                    PYTHONPATH=.  screen -dm  python methods/train_${method}.py --gpu ${gpu} --dataset ${dataset} --model ${model} --seed ${seed}  \
                    --save-path outputs/${dataset}/${method}_${model}/${seed} --learning_rate 3e-4 --adam ${extra_params}
                else
                    echo "Using SGD for ${method} on ${dataset} with ${model} and seed ${seed}"
                    PYTHONPATH=. screen -dm python methods/train_${method}.py --gpu ${gpu} --dataset ${dataset} --model ${model} --seed ${seed}  \
                        --save-path outputs/${dataset}/${method}_${model}_TMP/${seed} --learning_rate ${lr} ${extra_params}
                fi 

                gpu=$(((gpu+1)%4))

                # done

                # if [ $gpu -eq 0 ]; then
                #     gpu=1
                # fi
            done      
        done
    done
done

# PYTHONPATH=. screen -dm python methods/train_SGD.py --gpu 0 --dataset cifar10 --model vit-s-32
# PYTHONPATH=.  python methods/train_ExtrapSAM.py --gpu 0 --dataset cifar10 --model vit --adam --learning_rate 1e-4 --rho 0.02 --seed 0

# PYTHONPATH=.  python methods/train_ExtrapSAM.py --gpu 0 --dataset cifar10 --model resnet18