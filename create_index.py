import numpy as np
import faiss
import glob 
import argparse

# 这个脚本的主要功能是针对一系列保存有特征描述子的 .npy 文件，
# 生成一个 FAISS 索引，方便后续的相似度搜索。支持两种索引类型：
# - 'flat': 利用暴力搜索（IndexFlatL2），适用于较小规模数据。
# - 'ivf': 利用倒排索引（IndexIVFFlat），适合大规模数据，此时需要先进行训练。
#
# 除此之外，还提供了将 .npy 文件按照一定规则分组加载的功能，以防一次性加载所有文件占用过多内存。

def create_faiss_index_for_group(file_group, group_name, index=None, index_type='flat', nlist=100):
    """
    为一个文件组（file_group）创建/更新FAISS索引。

    参数：
    - file_group: 包含 .npy 文件路径的列表。每个文件存储了一部分描述子。
    - group_name: 当前文件组的名称，用于输出日志信息。
    - index: 上一个索引对象，如果为 None 则创建新的索引。
    - index_type: 索引的类型。'flat' 表示直接使用 L2 距离，'ivf' 表示倒排文件索引，需要先训练。
    - nlist: 仅当 index_type 为 'ivf' 时使用，表示倒排列表的数量（聚类个数）。

    功能：
    1. 遍历 file_group 中的每个文件，加载并收集描述子，然后将它们拼接到一个大的数组 descriptors 中。
    2. 如果没有提供现有的 index，则根据 index_type 初始化索引：
       - 如果 index_type 为 'flat'，使用 IndexFlatL2。
       - 如果 index_type 为 'ivf'，使用 IndexIVFFlat，并首先训练索引。
    3. 将所有描述子添加到索引中。
    4. 输出当前索引中向量的数量，并返回更新后的索引对象。
    """
    arrays = [] 
    # 加载 file_group 中的所有 .npy 文件
    for file_path in file_group:
        descriptors = np.load(file_path)  # 从 .npy 文件加载描述子
        arrays.append(descriptors)
        dimension = descriptors.shape[-1]  # 获取描述子的维度（假定每个文件拥有相同的维度）
    
    # 将所有文件中的描述子合并成一个大的数组
    descriptors = np.concatenate(arrays, axis=0)
    
    # 如果未传入现有索引，则新建一个索引
    if index is None:
        if index_type == 'flat':
            # 使用暴力搜索索引，直接计算 L2 距离
            index = faiss.IndexFlatL2(dimension)
        elif index_type == 'ivf':
            # 构造量化器（这里也使用 IndexFlatL2）用于构建 IVF 索引
            quantizer = faiss.IndexFlatL2(dimension)
            # 创建IVF索引：nlist 表示聚类中心的数量
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            # 训练索引，这一步需要提供数据
            index.train(descriptors)
        else:
            raise ValueError("Unsupported index type specified.")

    # 将当前组的描述子添加到索引中
    index.add(descriptors)

    # 输出当前组添加后的向量数量，以便确认是否添加成功
    print(f'Group {group_name}: Number of vectors in the index: {index.ntotal}')
    return index

def generate_groups(npy_directory, num_groups):
    """
    将指定目录中的 .npy 文件进行分组。

    参数：
    - npy_directory: .npy 文件存放的目录路径。
    - num_groups: 需要分为多少组，即按照组的数量均匀划分文件。

    功能：
    1. 利用 glob 模块搜集目录下的所有 .npy 文件，并排序。
    2. 通过切片操作，将文件列表分成 num_groups 个子列表。

    返回值：
    - 包含文件组的列表，每个子列表中存放了一部分文件路径。
    """
    # 查找并排序所有目录中的 .npy 文件
    all_files = sorted(glob.glob(npy_directory + '/*.npy'))
    
    groups = []
    # 使用切片技巧将文件列表均匀分成 num_groups 个部分
    for i in range(num_groups):
        group = all_files[i::num_groups]  # 从第 i 个元素开始，每隔 num_groups 个取一个
        groups.append(group)

    return groups

def parse_args():
    """
    解析命令行参数，根据用户输入设置 FAISS 索引的参数。

    可选参数：
    --npy_directory: 保存 .npy 文件的目录路径。
    --index_file: 存储生成的 FAISS 索引的目标文件路径。
    --index_type: 指定创建索引的方法，可选值为 'flat' 或 'ivf'（默认为 'flat'）。
    --group_num: 将 .npy 文件分为多少组；group_num=1 表示一次加载所有文件。

    返回值：
    - 包含解析后参数的对象。
    """
    parser = argparse.ArgumentParser(
        description="Create FAISS index from descriptor files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--npy_directory", type=str, required=True, help="Directory containing .npy files with descriptors")
    parser.add_argument("--index_file", type=str, required=True, help="Output file path to save the FAISS index")
    parser.add_argument("--index_type", type=str, default='flat', choices=['flat', 'ivf'],
                        help="Method of creating the index: 'flat' for IndexFlatL2, 'ivf' for IndexIVFFlat")
    parser.add_argument('--group_num', type=int, default=1, 
                        help='Number of groups to partition the .npy files. group_num=1 means all files are loaded together.') 
    args = parser.parse_args()
    return args

if __name__ == '__main__': 
    # 解析命令行参数
    args = parse_args()
    
    # 将 .npy 文件按照指定的组数进行分组
    file_groups = generate_groups(args.npy_directory, args.group_num)
    
    index = None  # 起始时未创建索引
    # 依次处理每个文件组，将各组的描述子逐步添加到索引中
    for i, file_group in enumerate(file_groups):
        group_name = f'group_{i+1}'
        index = create_faiss_index_for_group(file_group, group_name, index=index,
                                             index_type=args.index_type)
    
    # 将构建好的 FAISS 索引写入指定的文件中
    faiss.write_index(index, args.index_file)
    print(f'Saved index to {args.index_file}')