wei_arr=(
0.0783
0.0620
0.0493
0.0394
0.0315
0.0252
0.0201
0.0161
0.0129
0.0103)

adj_arr=(
0.5656
0.5372
0.5102
0.4843
0.4598
0.4368
0.4149
0.3941
0.3737
0.3547)


for model in gcn
do
    for data in cora citeseer pubmed Cornell Texas Wisconsin Computers Photo
    do
        for i in ${!wei_arr[@]}
        do
            for pr in 0.1 0.2 0.3
            do
                for uf in 10 20 30
                do
                    for fpe in 50 100 150
                    do
                        python main_stgnn.py --method GraNet \
                                                --prune-rate $pr \
                                               --optimizer adam \
                                               --sparse-init ERK \
                                               --init-density 1.0 \
                                               --final-density ${wei_arr[$i]} \
                                               --final-density_adj ${adj_arr[$i]} \
                                               --final-density_feature 1.0 \
                                               --update-frequency $uf \
                                               --l2 0.0005 \
                                               --lr 0.01 \
                                               --epochs 200 \
                                               --model $model \
                                               --data $data \
                                               --final-prune-epoch $fpe \
                                               --growth_schedule momentum  \
                                               --adj_sparse \
                                               --weight_sparse \
                                               --sparse
                    done
                done
            done
        done
    done
done





# --model: gcn, gat, sgc, appnp, gcnii (5)
# --data: cora, citeseer, citeseer, Cornell, Texas, Wisconsin, Actor
# --data: CS, Physics, Computers, Photo, WikiCS, reddit
# --data: ogbn-arxiv, ogbn-proteins, ogbn-products, ogbn-papers100M (17)
# --weight_sparse or --feature_sparse --sparse (7)
# --sparse: base or sparse train (2)

# --method: GraNet, GraNet_uniform, GMP, GMP_uniform (4)
# --growth_schedule: gradient, momentum, random (3)
# --sparse_init: uniform, ERK (2)

# --prune-rate : regenration rate : 0.1, 0.2, 0.3  (3)
# --update-frequency 10 20 30 (3)
# --final-prune-epoch 50 100 150 (3)

# --init-density:  weight init density: 1, (dense to sparse)
# --final-density: weight : 0.5 0.1 0.01 0.0001 (4)
# --final-density_adj : 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.01 (10)
# --final-density_feature: 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.01 (10)


#  5 x 17 x 7  x 4 x  3 x 2  X 3 X 3 X 3 X 4 X 10 X 10 = 154,224,000

# Actual: 4 x 13 x 4 x 10 x 3 x 3 x 3 = 56160


# python main_stgnn.py --method GraNet \
#                --prune-rate 0.5 \
#                --optimizer adam \
#                --sparse-init ERK \
#                --init-density 0.5 \
#                --final-density 0.1 \
#                --update-frequency 10 \
#                --l2 0.0005 \
#                --lr 0.01 \
#                --epochs 200 \
#                --model gcn \
#                --data cora \
#                --final-prune-epoch 100