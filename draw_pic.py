import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
from lib.toy_data import generate_slope, generate_swissroll3d
from lib.visualize_flow import plt_samples, set_figure, standard_fig_save
import os
import umap
from sklearn import manifold
from lib.WLLE import WLLE
import time


def get_color(color='rgb', data_set='left'):
    """

    Parameters
    ----------
    color = rgb or height

    Returns color matrix of shape=(num_samples, RGB_channels)
    -------

    """
    file = r"D:\projects\SF\toy_example\data\{}.csv".format(data_set)
    data = np.loadtxt(fname=file, skiprows=1, delimiter=",")[:, 0:3]
    if color == 'rgb':
        # using original color as the rgb tuple
        colors = np.zeros((3, data.shape[0]))
        for i in range(3):
            h = data[:, i]
            temp = (h - h.min()) / (h.max() - h.min())
            # temp = (h) / (h.max() - h.min())
            # temp = (h - np.mean(h, axis=0)) / np.std(h, axis=0)
            colors[i] = temp
        colors = colors.T
    elif color == 'height':
        height = data[:, 2]
        min_hei, max_hei = np.min(height), np.max(height)
        colors = [((i - min_hei) * 1.0) / (max_hei - min_hei) * 256 for i in height]
        # colors = colors.T
        # colors = np.tile(colors, (3)).reshape([len(colors), -1])
    return colors


def get_manifold_data(mode='HLLE', data_set='DDH_left', k=50, normalization='max_min', color='rgb',
                      shuffle=True, save_path=None, date=4):
    # 绘制manifold的效果，散点图
    ori_path = r"D:\projects\Datasets\Original\{}.csv".format(data_set)
    ori_data = np.loadtxt(fname=ori_path, skiprows=1, delimiter=",")[:, 0:3]
    d_data = np.loadtxt(fname=ori_path, skiprows=1, delimiter=",")[:, date:date+1]
    train_data = np.concatenate((ori_data, d_data), axis=1)

    # normalization, 经度，维度，海拔
    colors = np.zeros((3, ori_data.shape[0]))

    for i in range(train_data.shape[1]):
        h = train_data[:, i]
        temp = (h - h.min()) / (h.max() - h.min())
        train_data[:, i] = temp
        if i < 3:
            h = ori_data[:, i]
            temp = (h - h.min()) / (h.max() - h.min())
            colors[i] = temp
    colors = colors.T

    # manifold learning
    if mode == "LLE":
        trans_data = manifold.LocallyLinearEmbedding(n_neighbors=k, n_components=2,
                                                     method='standard').fit_transform(train_data)
    elif mode == "HLLE":
        trans_data = manifold.LocallyLinearEmbedding(n_neighbors=k, n_components=2,
                                                     method='hessian').fit_transform(train_data)
    elif mode == "MLLE":
        trans_data = manifold.LocallyLinearEmbedding(n_neighbors=k, n_components=2,
                                                     method='modified').fit_transform(train_data)
    elif mode == "LTSA":
        trans_data = manifold.LocallyLinearEmbedding(n_neighbors=k, n_components=2,
                                                     method='ltsa').fit_transform(train_data)
    elif mode == "T-SNE":
        trans_data = manifold.TSNE(n_components=2, init='pca', random_state=77).fit_transform(train_data)
    elif mode == 'Isomap':
        trans_data = manifold.Isomap(n_neighbors=k, n_components=2).fit_transform(train_data)
    elif mode == 'MDS':
        trans_data = manifold.MDS(n_components=2).fit_transform(train_data)
    elif mode == 'SE':
        trans_data = manifold.SpectralEmbedding(n_components=2).fit_transform(train_data)
    elif mode == 'UMAP':
        reducer = umap.UMAP(n_neighbors=k, n_components=2, metric='euclidean', random_state=42)
        reducer.fit(train_data)
        trans_data = reducer.transform(train_data)
    elif mode == 'WLLE':
        trans_data = WLLE(train_data, n_neighbors=k, n_components=2, gamma=2)
    else:
        trans_data = train_data

    if save_path:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = "{}_{}_{}_{}".format(data_set, mode, k, date)
        np.savetxt(save_path + "/{}.csv".format(file_name), trans_data, delimiter=',', fmt='%.32e')
        # show_manifold_simple(save_path + "/{}.csv".format(file_name), data_set=data_set, img_name=file_name)

        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.scatter(trans_data[:, 0], trans_data[:, 1], marker='o', c=colors, s=25)
        plt.savefig(save_path + "/{}.pdf".format(file_name), dpi=500, bbox_inches='tight', pad_inches=0)
        # plt.show()
        plt.cla()
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.scatter(trans_data[:, 0], trans_data[:, 1], marker='o', c=colors, s=25)
        plt.savefig(save_path + "/{}.png".format(file_name), dpi=500, bbox_inches='tight', pad_inches=0)
        # plt.show()
        plt.cla()
    return trans_data


def get_color_from_data(data):
    colors = np.zeros((3, data.shape[0]))
    for i in range(3):
        h = data[:, i]
        temp = (h - h.min()) / (h.max() - h.min())
        # temp = (h) / (h.max() - h.min())
        # temp = (h - np.mean(h, axis=0)) / np.std(h, axis=0)
        colors[i] = temp
    colors = colors.T
    return colors


def draw_three_d_scatter(data, color=None):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, aspect='auto', projection='3d')
    plt_samples(data, ax, title=None)

    plt.show()
    standard_fig_save("results/pics/", "3d_scatter")


def draw_three_d_scatter_from_file(dataset='left', color='rgb'):
    file = "data/use_info_{}.csv".format(dataset)
    train_data = np.loadtxt(fname=file)[:, 0:3]
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    draw_three_d_scatter(train_data, color)


def draw_two_d_scatter(dataset="2", color='rgb'):
    file = "data/YC0{}_rel.csv".format(dataset)
    train_data = np.loadtxt(fname=file, delimiter=',', skiprows=1)[:, 0:3]
    x = train_data[:, 0]
    y = train_data[:, 1]
    z = train_data[:, 2]

    # file = "data/YC0{}_rel.csv".format(1)
    # train_data = np.loadtxt(fname=file, delimiter=',', skiprows=1)[:z.shape[0], 0:3]
    # ly = train_data[:, 1]

    maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    for cm in maps:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), tight_layout=True)
        plt.axis('off')
        ax.scatter(x, y, s=15, c=z, marker='o', cmap=cm)
        plt.savefig("results/2d/2d_origin_{}_{}.pdf".format(dataset, cm), dpi=500, bbox_inches='tight',
                    transparent=True,
                    pad_inches=0)
    # plt.show()


def his_scatter_draw(dataset, color='rgb'):
    # 为了避免遮挡，可以使用流形
    if dataset == 1 or dataset == 2:
        data_path = r"D:\projects\SF\toy_example\data\sxg"
        # dataset = 1
        abs_name = "YC0{}_abs".format(dataset)
        rel_name = "YC0{}_rel".format(dataset)
        file_name = os.path.join(data_path, abs_name + ".csv")
        cor = np.loadtxt(fname=file_name, delimiter=',', skiprows=1)[:, 0:3]
        data = np.loadtxt(fname=file_name, delimiter=',', skiprows=1)[:, 3:]
    else:
        data_path = r"D:\projects\SF\toy_example\data"
        abs_name = "use_info_{}".format(dataset)
        rel_name = abs_name
        file_name = os.path.join(data_path, abs_name + ".csv")
        cor = np.loadtxt(fname=file_name, skiprows=1)[:, 0:3]
        data = np.loadtxt(fname=file_name, skiprows=1)[:, 3:]

    x = cor[:, 0]
    y = cor[:, 1]
    z = cor[:, 2]
    maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges',
            'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    cm = maps[1]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), tight_layout=True)
    ax.scatter(x, y, s=5, c=z, marker='o', cmap=cm)
    # standard_fig_save(save_path="results/2d", file_name="2d_origin", file_format="png")
    for date in range(data.shape[1]):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), tight_layout=True)
        ax.scatter(x, y, s=10, c=data[:, date], marker='o', cmap=cm)
        ax.set_xticks([])
        ax.set_yticks([])
        standard_fig_save(save_path="results/{}_history".format(dataset), file_name="{}_{}".format(dataset, date),
                          file_format="png")


def scatter2hot(X, n, cmapid):
    # X 是输入的散点，前2是坐标，第三是值，n是其展示的粒度。转换到一个方阵中
    normalization = 'max_min'
    for i in [0, 1, 2]:
        h = X[:, i]
        if normalization == 'max_min':
            temp = (h - h.min()) / (h.max() - h.min())
        X[:, i] = temp

    x_max = X[:, 0].max()
    x_min = X[:, 0].min()
    y_max = X[:, 1].max()
    y_min = X[:, 1].min()
    x_in = (x_max - x_min) / n
    y_in = (y_max - y_min) / n
    x_list = np.arange(x_min, x_max, x_in)
    y_list = np.arange(y_min, y_max, y_in)
    squre = np.empty((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            temp = X
            temp = temp[x_list[i] < temp[:, 0]]
            if i < n - 1: temp = temp[temp[:, 0] < x_list[i + 1]]
            temp = temp[y_list[j] < temp[:, 1]]
            if j < n - 1: temp = temp[temp[:, 1] < y_list[j + 1]]

            if temp.shape[0] == 0:
                squre[i][j] = -10
            else:
                squre[i][j] = temp[:, 2].mean()
    # 逆时针旋转90
    squre = np.rot90(squre, 1)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = plt.subplot()
    maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens',
            'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu',
            'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
    im = ax.imshow(squre, cmap=maps[cmapid])
    ax.set_xticks([])
    ax.set_yticks([])
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig("results/hot/{}_{}.pdf".format(n, cmapid), quality=100, dpi=500, bbox_inches='tight',transparent=True,
                pad_inches=0)


if __name__ == '__main__':
    # his_scatter_draw(dataset="left")

    plt.rcParams['ps.useafm'] = True
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    # show manifold shape
    datasets = ['YC01_rel', 'YC02_rel', "DDH_left", "DDH_right"]

    a = ['MLLE', 'Isomap', 'T-SNE', 'UMAP']
    b = ['HLLE', 'LTSA']
    c = ['WLLE']
    e = ['LLE']

    # for mode in ['LTSA']:
    #     for dataset in ["DDH_right"]:
    #         print(dataset, mode)
    #         get_manifold_data(mode, dataset, save_path=r"D:\projects\Datasets\experiments\{}".format(mode), k=150)

    # for date in [5, 10, 15, 20]:
    #     for mode in c:
    #         for dataset in datasets:
    #             print(dataset, mode, date)
    #             get_manifold_data(mode, dataset, save_path=r"D:\projects\Datasets\experiments\{}".format(mode), k=150, date=date)
    #             time.sleep(10)

    for date in [40, 50]:
        for k in range(400, 650, 50):
            for dataset in ["DDH_right"]:
                mode = "WLLE"
                print(dataset, mode, date)
                get_manifold_data(mode, dataset, save_path=r"D:\projects\Datasets\experiments\{}".format(mode), k=k,
                                  date=date)
                time.sleep(10)

    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(1, 1, 1, aspect='auto', projection='3d')
    #
    # # samples = generate_slope(n=10000, f=2, save_path='data/random_slope.csv')
    # # samples = generate_swissroll3d(n_samples=10000, noise=0, save_path='data/swissroll3d.csv')
    # samples = 1
    # plt_samples(samples, ax)
    # plt.show()

