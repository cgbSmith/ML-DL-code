import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    #np.identity(n)==》创建n维单位矩阵
    #np.identity(len(classes))[i, :]即分别取单位矩阵的每一行
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    #map函数类似for循环，这里会遍历labels的每一项，然后执行class_dict.get函数，即根据标签，取字典中查找该标签对应的什么向量,map放回的是map object对象，需要用list转换为列表
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    #将content的特征矩阵用二维数组储存起来
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    #取二维数组的每一行，每一行切片保留[1:-1]的内容，即提取整个二维数组的第1列至倒数第二列。（第0列为编号，最后一列为分类名，这里只提取特征属性0、1）
    f_test =idx_features_labels[:, 1:-1]
    #将features矩阵转为稀疏矩阵
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    #提取二位数组最后一列，即分类名，经过onehot处理后，lables变为，每行都只有一个一，其他全为0的矩阵
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    # 取二维数组的第一列，即取每一条数据（节点）的编号31336，1061127。。。
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    #给每一个节点对应进行再编号，比如节点31336对应数组的第0个，编号1061127的节点对应第1个。。。
    idx_map = {j: i for i, j in enumerate(idx)}
    #将cora.cites中边的关系用数组存起来，比如35与1033有边，就存为[35,1033]，or就是将cora.cites信息读入
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # add to monitor
    #将edges_unordered数据展成一行
    add_monitor = edges_unordered.flatten()
    ### end add to monitor
    #将边的关系用之前节点的编号表示，比如一条边需要用35，1033表示，那么这里就是用35和1033对应的编号进行表示
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    #这里用于创建邻接数组，edges.shape[0]表示有多少条边，那么对应的邻接数组中就有多少个1，，因此用np.ones(edges.shape[0]来创建对应数目的全是1的数组)
    #sp.coo_matrx的第三种用法为（查官网）：coo_matrix((data, (i, j)), [shape=(M, N)])，data为输入到矩阵的数据，i表示数据的行标号，j表示数据的列标号，即A[i[k],j[k]]=data[k]
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),#表示矩阵的形状是节点数*节点数，shape[0]表示有多少行
                        dtype=np.float32)

    # build symmetric adjacency matrix
    # multiply代表两个矩阵每一个对应元素相乘，如果两个矩阵的规模不一样那么就通过广播进行
    #将矩阵变为对称的邻接矩阵，即有向边变为无向边。
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)


    features = torch.FloatTensor(np.array(features.todense()))
    #np.where返回二维数组中不为0元素所在的行数组和列数组
    #np.where()[0]返回行数组,np.where()[1]返回列数组
    #这里的labels就是表示有多少该节点对应的分类的编号，一共有6
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    #power取每一个元素的-1次方
    #degree矩阵可以通过A矩阵行向量的和，再转成对角矩阵
    r_inv = np.power(rowsum, -1).flatten()
    #若有超过限制的就把它置为0
    r_inv[np.isinf(r_inv)] = 0.
    #diags函数将一n维数组转变为n维对角矩阵
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
    #最终需要的邻接矩阵公式为：A~ = D（-1）A

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)# max(1)返回每一行最大值组成的一维数组和索引,output.max(1)[1]表示最大值所在的索引indice
    correct = preds.eq(labels).double()  #equ()两者相等置1，不等值0
    correct = correct.sum() # 对其求和，即求出相等(置1)的个数
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # 先转换成coo_matrix形式的矩阵，因为Torch支持COO（rdinate）格式的稀疏张量
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    # np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组
    # 这一句是提取稀疏矩阵的非零元素的索引。得到的矩阵是一个[2, 8137]的tensor。
    # 其中第一行是行索引，第二行是列索引。每一列的两个值对应一个非零元素的坐标。
    #indices为coo矩阵的索引
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    #规定数值和shape
    #values为coo矩阵的值
    values = torch.from_numpy(sparse_mx.data)
    #shape为coo矩阵的形状大小
    shape = torch.Size(sparse_mx.shape)
    print(shape)
    return torch.sparse.FloatTensor(indices, values, shape)

    ###sparse_mx_to_torch_sparse_tensor函数总结：
    # 总的来说，这个函数先将csr_matrix矩阵转为coo_matrix矩阵，先后获取coo矩阵的中非零元素的行列索引号
    # 然后获取coo矩阵中非零元素的值（邻接矩阵经过归一化，基本都是小数），之后在获取coo矩阵的形状
    # 然后通过toch.sparse.FloatTensor函数通过行列索引和值和形状，然后做出tensor类型的稀疏矩阵
