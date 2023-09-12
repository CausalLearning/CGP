for model in gcn gat sgc appnp
do
    for data in cora citeseer pubmed Cornell Texas Wisconsin Actor CS Physics Computers Photo ogbn-arxiv
    do
        for fde in 0.5 0.1 0.01
        do
            for fdf in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.01
            do
                for fda in 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.01
                do
                    for pr in  0.1 0.2 0.3
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
                                                   --final-density $fde \
                                                   --final-density_adj $fda \
                                                   --final-density_feature $fdf \
                                                   --update-frequency $uf \
                                                   --l2 0.0005 \
                                                   --lr 0.01 \
                                                   --cuda $1 \
                                                   --epochs 200 \
                                                   --model $model \
                                                   --data $data \
                                                   --final-prune-epoch $fpe \
                                                   --growth_schedule momentum  \
                                                   --feature_sparse \
                                                   --weight_sparse \
                                                   --adj_sparse \
                                                   --sparse
                            done
                        done
                    done
                done
            done
        done
    done
done


# cora citeseer



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