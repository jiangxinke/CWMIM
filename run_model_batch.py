import os
run = r'python D:\projects\SF\toy_example\train.py '
flow = r'python D:\projects\SF\toy_example\flowone_train.py '
cnf_no_flow = r'python D:\projects\SF\toy_example\baselines\train_CNF.py '
GCN = r'python D:\projects\SF\toy_example\train_SIG-VAE.py '
cnf_no_s = r'python D:\projects\SF\toy_example\train_noS.py '
i = 0
# 注意相同的log key会覆盖之前的图片和储存的model

# 对比不同的manifold的区别，准备用箱型图来展示
iter = 100
log_key = "mani-"
i = 36
for manifold in ['normal', 'nflow', 'MLLE', 'Isomap', 'T-SNE', 'UMAP', 'HLLE', 'LTSA', 'LLE', 'WLLE']:
    for data in ['YC01_rel', 'YC02_rel', "DDH_left", "DDH_right"]:
    # for data in ['YC01_rel', 'YC02_rel']:
        i = i+2
        os.system(run + r"--val_freq=1 --lr=0.05 --log_key={} --niters={} --data={}  --manifold={} --conti=True --patient=100 --seed=1024 "
                        r"--save_file=D:\projects\SF\toy_example\results\流形对比数据3.csv".format(log_key + manifold, iter, data, manifold))

# 只要W不要S
# iters = 1000
# log_key = "no_s"
# for data in ['YC01_rel', 'YC02_rel', "DDH_left", "DDH_right"]:
#     os.system(cnf_no_s + "--log_key={} --niters={} --data={} --val_freq=50 --lr=0.05".format(log_key, iters, data))

# iters = 1000
# log_key = "GCN"
# for data in ['YC01_rel', 'YC02_rel', "DDH_left", "DDH_right"]:
#     os.system(GCN + "--log_key={} --niters={} --data={} --val_freq=10 --lr=0.05".format("GCN", iters, data))
#     os.system(GCN + "--log_key={} --niters={} --data={} --val_freq=10 --lr=0.05 --VGAE=True".format("VGAE", iters, data))

# W size 的影响测试
# iter = 10000
# for data in ['YC01_rel', "YC02_rel", "DDH_left", "DDH_right"]:
#     for leng in [1, 2, 3, 4, 5]:
#         os.system(run + "--log_key={} --niters={} --conti={} --VGAE={} --data={} --hidden_len={}".format("hidden-{}".format(leng), iter, False, False, data, leng))


# 对比生成的GCN和非生成的
# iter = 10000
# os.system(run + "--log_key={} --niters={} --conti={} --VGAE={}".format("gnn-VGAE", iter, False, True))
# os.system(run + "--log_key={} --niters={} --conti={} --VGAE={}".format("gnn-GCN", iter, False, False))




# get flow manifold--finished
# log_key = "yc01"
# flow1iter = 200
# seed = 1234
# for data in ['YC02_rel', "DDH_left", "DDH_right"]:
#     os.system(flow + "--num_blocks={} --dims={} --log_key={} --niters={} --data={} --lr={}".format(1, "64-64-64", log_key, flow1iter, data, 5e-3))
# run yc01
# os.system(flow + "--log_key={} --niters={} --data={} --lr={} --val_freq={} --special_design=True".format(log_key, flow1iter, 'YC01_rel', 9e-2, 10))

# 无W的CNF训练
# log_key = "cnf_no_W"
# flow1iter = 2000
# seed = 1234
# for data in ['YC01_rel', 'YC02_rel', "DDH_right"]:
#     os.system(cnf_no_flow + "--num_blocks={} --dims={} --log_key={} --niters={} --data={} --lr={}".format(1, "64-64-64", log_key, flow1iter, data, 5e-3))

# log_key = "uniform-gae"
# flow1iter = 50000
# flow2iter = 5000
# seed = 1234
# os.system(flow + "--num_blocks={} --dims={} --log_key={} --niters={} --conti={}".format(1, "64-64-64", log_key, 1, False))
# os.system(run + "--num_blocks={} --dims={} --log_key={} --niters={} --conti={}".format(1, "64-64-64", log_key, 1, False))
# os.system(flow + "--num_blocks={} --dims={} --log_key={} --niters={} --seed={} --VAGE={}".format(1, "64-64-64", log_key, flow1iter, seed + i*i, True))
# os.system(run + "--num_blocks={} --dims={} --log_key={} --niters={} --seed={} --VAGE={}".format(1, "64-64-64", log_key, flow1iter, seed + i*i, True))

# 标准分布+GCN
# log_key = "uniform"
# flow1iter = 50000
# flow2iter = 5000
# seed = 1234
# os.system(flow + "--num_blocks={} --dims={} --log_key={} --niters={} --conti={}".format(1, "64-64-64", log_key, 1, False))
# os.system(run + "--num_blocks={} --dims={} --log_key={} --niters={} --conti={}".format(1, "64-64-64", log_key, 1, False))
# os.system(flow + "--num_blocks={} --dims={} --log_key={} --niters={} --seed={}".format(1, "64-64-64", log_key, flow1iter, seed + i*i))
# os.system(run + "--num_blocks={} --dims={} --log_key={} --niters={} --seed={}".format(1, "64-64-64", log_key, flow1iter, seed + i*i))


# log_key = "FT3k2h10R_vgae"
# flow1iter = 3000
# flow2iter = 200
# ruoud = 10
# seed = 1234
# os.system(flow + "--num_blocks={} --dims={} --log_key={} --niters={} --conti={}".format(1, "64-64-64", log_key, 1, False))
# os.system(run + "--num_blocks={} --dims={} --log_key={} --niters={} --conti={}".format(1, "64-64-64", log_key, 1, False))
# for i in range(ruoud):
#     os.system(flow + "--num_blocks={} --dims={} --log_key={} --niters={} --seed={} --VAGE={}".format(1, "64-64-64", log_key, flow1iter, seed + i*i, True))
#     os.system(run + "--num_blocks={} --dims={} --log_key={} --niters={} --seed={} --VAGE={}".format(1, "64-64-64", log_key, flow1iter, seed + i*i, True))


# log_key = "fusion_train"
# flow1iter = 2000
# flow2iter = 200
# ruoud = 5
# for i in range(ruoud):
#     os.system(flow + "--num_blocks={} --dims={} --log_key={} --niters={} --seed={}".format(1, "64-64-64", log_key, flow1iter, seed))
#     os.system(run + "--num_blocks={} --dims={} --log_key={} --niters={} --seed={}".format(1, "64-64-64", log_key, flow1iter, seed))
# flow1不收敛，flow2形状每个R都及其相似，检查是否是seed的问题
# for i in range(10):
# os.system(flow + "--num_blocks={} --dims={} --log_key={} --niters={}".format(1, "64-64-64", "fusion_train", 500))
# os.system(run + "--num_blocks={} --dims={} --log_key={} --niters={}".format(1, "64-64-64", "fusion_train", 200))
# 在1R200e的时候几乎收敛，损失RMSE达到了0.098。看2R会不会有进一步的提升。2R


# 不需要多次实验，在seed固定的情况下，训练过程完全一致
# os.system(run + "--num_blocks={} --dims={} --log_key={}".format(2, "64-64-64", "block_2_dims_64"))
# os.system(run + "--num_blocks={} --dims={} --log_key={}".format(1, "128-128-128", "block_1_dims_128"))
# os.system(run + "--num_blocks={} --dims={} --log_key={}".format(2, "128-128-128", "block_2_dims_128"))
# os.system(run + "--num_blocks={} --dims={} --log_key={} --batch_size={}".format(2, "128-128-128", "block_2_dims_128_b8k", 8000))
# os.system(run + "--num_blocks={} --dims={} --log_key={} --batch_size={}".format(3, "128-128-128", "block_3_dims_128", 8000))
# os.system(run + "--num_blocks={} --dims={} --log_key={} --batch_size={}".format(4, "128-128-128", "block_4_dims_128", 8000))
# os.system(run + "--num_blocks={} --dims={} --log_key={} --batch_size={}".format(5, "128-128-128", "block_5_dims_128", 8000))

# 全随机数据集
# os.system(run + "--num_blocks={} --dims={} --log_key={}".format(1, "64-64-64", "block_2_dims_64"))
# os.system(run + "--num_blocks={} --dims={} --log_key={}".format(1, "128-128-128", "block_2_dims_64"))
# os.system(run + "--num_blocks={} --dims={} --log_key={}".format(5, "256-256-256", "block_5_dims_256"))

# os.system(run + "--num_blocks={} --dims={} --log_key={} --batch_size={}".format(3, "128-128-128", "block_3_dims_128", 8000))

# eps的分布影响
# os.system(run + "--data={} --input_dim={} --num_blocks={} --dims={} --log_key={} --random_type={}".format("2spirals_1d", 2, 5, "256-256-256", "uni-5", "uni"))
# os.system(run + "--data={} --input_dim={} --num_blocks={} --dims={} --log_key={} --random_type={}".format("2spirals_1d", 2, 5, "256-256-256", "nor-5", "nor"))
# os.system(run + "--data={} --input_dim={} --num_blocks={} --dims={} --log_key={} --random_type={}".format("2spirals_1d", 2, 1, "64-64-64", "nor-1-64", "nor"))
# os.system(run + "--data={} --input_dim={} --num_blocks={} --dims={} --log_key={} --random_type={}".format("2spirals_1d", 2, 1, "64-64-64", "zero-uni-1-64", "uni"))
# 研究std in 到底有没有用
# os.system(run + "--data={} --input_dim={} --num_blocks={} --dims={} --log_key={} --random_type={} --patient={}".format(
#     "2spirals_1d", 2, 1, "64-64-64", "std-in-1-64", "std_in", 500))
# os.system(run + "--data={} --input_dim={} --num_blocks={} --dims={} --log_key={} --random_type={} --patient={}".format(
#     "2spirals_1d", 2, 1, "64-64-64", "zero-in-1-64", "zero_in", 500))

# eps_g or not
# os.system(run + "--data={} --input_dim={} --num_blocks={} --dims={} --log_key={} --patient={} --eps_g={}".format(
#     "2spirals_1d", 2, 1, "64-64-64", "spirals_standard-1-64", 300, True))

# 使得eps更像一个正态分布
# os.system(run + "--data={} --input_dim={} --num_blocks={} --dims={} --log_key={} --patient={} --eps_g={} --std_min={} --std_max={}".format(
#     "2spirals_1d", 2, 1, "64-64-64", "eps-0.8-1.2-2spirals_1d-1-64", 300, False, 0.8, 1.2))

# os.system(run + "--data={} --input_dim={} --num_blocks={} --dims={} --log_key={} --patient={} --eps_g={} --conti={} --std_min={} --std_max={}".format(
#     "2spirals_1d", 2, 1, "64-64-64", "eps-0.1-0.3-2spirals_1d-1-64", 300, False, True, 0.1, 0.3))
# os.system(run + "--data={} --input_dim={} --num_blocks={} --dims={} --log_key={} --patient={} --eps_g={}".format(
#     "circles_1d", 2, 1, "64-64-64", "circles_standard-1-64", 300, False))
#
# os.system(run + "--data={} --input_dim={} --num_blocks={} --dims={} --log_key={} --patient={} --eps_g={}".format(
#     "2spirals_1d", 2, 1, "64-64-64", "spirals_epsg-1-64", 300, True))
# os.system(run + "--data={} --input_dim={} --num_blocks={} --dims={} --log_key={} --patient={} --eps_g={}".format(
#     "circles_1d", 2, 1, "64-64-64", "circles_epsg-1-64", 300, True))




