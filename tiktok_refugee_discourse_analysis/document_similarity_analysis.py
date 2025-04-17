import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, TruncatedSVD
import spacy
import jieba
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from wordcloud import WordCloud
from collections import Counter

# 加载英文Spacy模型
try:
    nlp_en = spacy.load("en_core_web_sm")
    print("英文Spacy模型加载成功")
except OSError:
    print("警告: 英文Spacy模型未找到，请先安装")
    print("pip install spacy")
    print("python -m spacy download en_core_web_sm")
    exit(1)

# 加载停用词
def load_stopwords(cn_path="chinese_stopwords.txt", en_path=None):
    """加载中英文停用词"""
    # 中文停用词
    try:
        with open(cn_path, 'r', encoding='utf-8') as f:
            cn_stopwords = set(f.read().splitlines())
    except FileNotFoundError:
        print(f"中文停用词文件 {cn_path} 未找到，使用空集合")
        cn_stopwords = set()
    
    # 英文停用词
    en_stopwords = set(nlp_en.Defaults.stop_words)
    if en_path:
        try:
            with open(en_path, 'r', encoding='utf-8') as f:
                custom_stopwords = set(f.read().splitlines())
                en_stopwords.update(custom_stopwords)
        except FileNotFoundError:
            print(f"英文停用词文件 {en_path} 未找到，只使用Spacy默认停用词")
    
    return cn_stopwords, en_stopwords

# 文本处理函数
def extract_text_from_comments(comments):
    """从评论中提取文本内容"""
    if not comments:
        return ""
    
    texts = []
    
    if isinstance(comments, list):
        for item in comments:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict) and 'text' in item:
                texts.append(item['text'])
    elif isinstance(comments, str):
        texts.append(comments)
    
    return " ".join(texts)

def preprocess_cn_text(text, stopwords):
    """预处理中文文本"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 去除标点和数字
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # 使用jieba分词
    words = jieba.lcut(text)
    
    # 过滤停用词和空词
    words = [word for word in words if word.strip() and word not in stopwords]
    
    return " ".join(words)

def preprocess_en_text(text, stopwords):
    """预处理英文文本"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 小写转换
    text = text.lower()
    
    # 去除标点和数字
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # 使用Spacy处理
    doc = nlp_en(text)
    
    # 词形还原并去除停用词
    tokens = []
    for token in doc:
        lemma = token.lemma_ if token.lemma_ != "-PRON-" else token.text
        if not token.is_stop and lemma not in stopwords and token.is_alpha and len(token.text) > 1:
            tokens.append(lemma)
    
    return " ".join(tokens)

def process_json_folder(folder_path, cn_stopwords, en_stopwords):
    """处理JSON文件夹，提取并预处理中英文评论"""
    cn_data = []
    en_data = []
    
    print("读取JSON文件...")
    # 遍历文件夹中的所有JSON文件
    for root, _, files in os.walk(folder_path):
        for file in tqdm(files, desc="处理文件"):
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                try:
                    # 读取JSON文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 处理中文评论
                    if 'comments_CN' in data:
                        text = extract_text_from_comments(data['comments_CN'])
                        processed_text = preprocess_cn_text(text, cn_stopwords)
                        if processed_text.strip():  # 只添加非空文本
                            cn_data.append({
                                'file': file,
                                'processed_text': processed_text,
                                'original_path': file_path
                            })
                    
                    # 处理英文评论
                    if 'comments_NonCN' in data:
                        text = extract_text_from_comments(data['comments_NonCN'])
                        processed_text = preprocess_en_text(text, en_stopwords)
                        if processed_text.strip():  # 只添加非空文本
                            en_data.append({
                                'file': file,
                                'processed_text': processed_text,
                                'original_path': file_path
                            })
                    
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    # 转换为DataFrame
    cn_df = pd.DataFrame(cn_data)
    en_df = pd.DataFrame(en_data)
    
    print(f"共处理 {len(cn_df)} 个包含中文评论的文件和 {len(en_df)} 个包含英文评论的文件")
    
    return cn_df, en_df

def calculate_cosine_similarity(df, min_df=2, max_df=0.9):
    """计算文本之间的余弦相似度"""
    if len(df) < 2:
        print("有效文本不足，无法计算相似度")
        return None, None, None
    
    print("计算TF-IDF向量...")
    # 创建TF-IDF向量化器
    vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
    
    # 转换文本为TF-IDF矩阵
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
    
    print("计算余弦相似度...")
    # 计算所有文档之间的余弦相似度
    cosine_sim = cosine_similarity(tfidf_matrix)
    
    # 创建相似度DataFrame以便查看
    files = df['file'].tolist()
    cosine_df = pd.DataFrame(cosine_sim, index=files, columns=files)
    
    return cosine_df, df, tfidf_matrix

def find_most_similar_pairs(cosine_df, threshold=0.4, top_n=20):
    """找出最相似的文档对"""
    # 创建包含所有相似度对的列表
    sim_pairs = []
    
    # 上三角矩阵（避免重复和自身比较）
    for i in range(len(cosine_df.index)):
        for j in range(i+1, len(cosine_df.columns)):
            sim = cosine_df.iloc[i, j]
            if sim >= threshold:
                sim_pairs.append((
                    cosine_df.index[i], 
                    cosine_df.columns[j], 
                    sim
                ))
    
    # 按相似度排序
    sim_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # 返回前top_n个结果
    return sim_pairs[:top_n]

# 可视化函数

# def plot_similarity_network(cosine_df, title, threshold=0.4, output_file=None, max_nodes=100):
#     """绘制相似度网络图"""
#     # 创建网络
#     G = nx.Graph()
    
#     # 添加节点
#     for file in cosine_df.index:
#         G.add_node(file)
    
#     # 添加边 (仅添加超过阈值的相似度)
#     for i in range(len(cosine_df.index)):
#         for j in range(i+1, len(cosine_df.columns)):
#             sim = cosine_df.iloc[i, j]
#             if sim >= threshold:
#                 G.add_edge(
#                     cosine_df.index[i],
#                     cosine_df.columns[j],
#                     weight=sim
#                 )
    
#     # 如果网络太大，提取最大的连通分量
#     if len(G.nodes) > max_nodes:
#         largest_cc = max(nx.connected_components(G), key=len)
#         G = G.subgraph(largest_cc).copy()
#         if len(G.nodes) > max_nodes:
#             # 还是太大，取前max_nodes个节点
#             sorted_nodes = sorted(G.nodes, key=lambda n: G.degree(n), reverse=True)[:max_nodes]
#             G = G.subgraph(sorted_nodes).copy()
    
#     # 设置节点大小 (基于度中心性)
#     degrees = dict(G.degree())
#     node_size = [v * 50 for v in degrees.values()]
    
#     # 设置边宽度 (基于权重)
#     edge_width = [G[u][v]['weight'] * 2 for u, v in G.edges()]
    
#     # 计算布局
#     pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
#     plt.figure(figsize=(12, 12))
    
#     # 绘制节点
#     nx.draw_networkx_nodes(G, pos, node_size=node_size, alpha=0.7,
#                           node_color=list(degrees.values()), cmap=plt.cm.viridis)
    
#     # 绘制边
#     nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.5, edge_color='#666666')
    
#     # 绘制标签 (仅度数最高的节点)
#     top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:15]
#     labels = {node: node for node, degree in top_nodes}
#     nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
    
#     plt.title(f"{title} - 相似度网络图 (阈值: {threshold})", fontsize=15)
#     plt.axis('off')
    
#     if output_file:
#         plt.savefig(output_file, dpi=300, bbox_inches='tight')
#     else:
#         plt.show()
    
#     plt.close()
    
#     return G  # 返回图对象，以便进一步分析

def plot_dimensionality_reduction(tfidf_matrix, df, title, method='pca', output_file=None):
    """使用降维技术可视化文档相似性"""
    # 选择降维方法
    if method.lower() == 'pca':
        # PCA适用于稠密矩阵
        dense_matrix = tfidf_matrix.toarray()
        reducer = PCA(n_components=2, random_state=42)
        reduced_data = reducer.fit_transform(dense_matrix)
        method_name = 'PCA'
    else:
        # 截断SVD适用于稀疏矩阵
        reducer = TruncatedSVD(n_components=2, random_state=42)
        reduced_data = reducer.fit_transform(tfidf_matrix)
        method_name = 'Truncated SVD'
    
    # 创建散点图
    plt.figure(figsize=(12, 10))
    
    # 绘制点
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.7, s=50)
    
    # 添加文件名作为标签 (仅添加部分点的标签，避免拥挤)
    if len(df) <= 20:
        # 如果点数量少，显示所有标签
        for i, file in enumerate(df['file']):
            plt.annotate(file, (reduced_data[i, 0], reduced_data[i, 1]), 
                        fontsize=8, alpha=0.8, ha='right')
    else:
        # 如果点数量多，只显示离中心远的点
        distances = np.sum(reduced_data**2, axis=1)
        top_indices = np.argsort(distances)[-20:]  # 最远的20个点
        for i in top_indices:
            plt.annotate(df['file'].iloc[i], (reduced_data[i, 0], reduced_data[i, 1]), 
                        fontsize=8, alpha=0.8, ha='right')
    
    plt.title(f"{title} - {method_name}降维可视化", fontsize=15)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


# def analyze_clusters(G, df):
#     """分析网络中的聚类并返回每个聚类的关键词"""
#     # 寻找社区 (使用Louvain方法)
#     try:
#         from community import community_louvain
#         partition = community_louvain.best_partition(G)
        
#         # 将节点归类到对应社区
#         communities = {}
#         for node, community_id in partition.items():
#             if community_id not in communities:
#                 communities[community_id] = []
#             communities[community_id].append(node)
        
#         # 只保留较大的社区
#         min_size = 3  # 最小社区大小
#         large_communities = {k: v for k, v in communities.items() if len(v) >= min_size}
        
#         # 分析每个社区的关键词
#         community_keywords = {}
#         for comm_id, nodes in large_communities.items():
#             # 获取该社区所有文档
#             comm_texts = []
#             for node in nodes:
#                 text = df.loc[df['file'] == node, 'processed_text'].values
#                 if len(text) > 0:
#                     comm_texts.extend(text)
            
#             # 提取关键词 (简单使用词频)
#             if comm_texts:
#                 all_text = " ".join(comm_texts)
#                 words = all_text.split()
#                 word_freq = Counter(words)
#                 # 取频率最高的15个词
#                 top_words = word_freq.most_common(15)
#                 community_keywords[comm_id] = top_words
        
#         return large_communities, community_keywords
#     except ImportError:
#         print("未安装python-louvain库，无法进行社区检测")
#         print("可以使用以下命令安装: pip install python-louvain")
#         return None, None

def main():
    # 获取输入路径
    json_folder = input("请输入JSON文件所在的文件夹路径: ")
    output_dir = input("请输入结果保存目录 (按Enter使用当前目录): ") or "."
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载停用词
    cn_stopwords_path = input("请输入中文停用词文件路径 (按Enter使用默认'chinese_stopwords.txt'): ") or "chinese_stopwords.txt"
    en_stopwords_path = input("请输入英文停用词文件路径 (可选，按Enter跳过): ")
    
    # 字体路径(用于中文词云)
    font_path = input("请输入中文字体文件路径 (可选，按Enter跳过): ")
    
    # 相似度参数
    threshold = float(input("请输入相似度阈值 (0-1，默认0.5): ") or "0.5")
    
    # 加载停用词
    cn_stopwords, en_stopwords = load_stopwords(cn_stopwords_path, en_stopwords_path)
    
    # 处理JSON文件
    cn_df, en_df = process_json_folder(json_folder, cn_stopwords, en_stopwords)
    
    # 处理中文评论
    if len(cn_df) >= 2:
        print("\n===== 分析中文评论 =====")
        cn_cosine_df, cn_processed_df, cn_tfidf_matrix = calculate_cosine_similarity(cn_df)
        
        # 找出最相似的文档对
        cn_similar_pairs = find_most_similar_pairs(cn_cosine_df, threshold)
        print(f"\n中文评论中最相似的 {len(cn_similar_pairs)} 个文档对:")
        for file1, file2, sim in cn_similar_pairs:
            print(f"{file1} 和 {file2}: {sim:.4f}")
        
        # 保存结果
        cn_output_file = os.path.join(output_dir, "chinese_cosine_similarity.csv")
        cn_cosine_df.to_csv(cn_output_file)
        print(f"中文评论相似度矩阵已保存到 {cn_output_file}")
        
        # 保存详细的相似对结果
        cn_pairs_output = os.path.join(output_dir, "chinese_similar_pairs.csv")
        with open(cn_pairs_output, 'w', encoding='utf-8') as f:
            f.write("File1,File2,Similarity\n")
            for file1, file2, sim in cn_similar_pairs:
                f.write(f"{file1},{file2},{sim:.6f}\n")
        print(f"中文评论最相似文档对已保存到 {cn_pairs_output}")
        
        # 可视化
        print("\n生成中文评论可视化...")
        
        
        # # 网络图
        # cn_graph = plot_similarity_network(cn_cosine_df, "中文评论", threshold,
        #                                 os.path.join(output_dir, "chinese_network.png"))
        
        # 降维可视化
        plot_dimensionality_reduction(cn_tfidf_matrix, cn_df, "中文评论", 
                                    output_file=os.path.join(output_dir, "chinese_pca.png"))
        
        
        # # 分析聚类
        # if cn_graph:
        #     cn_communities, cn_keywords = analyze_clusters(cn_graph, cn_df)
        #     if cn_communities:
        #         print("\n中文评论中的社区分析:")
        #         for comm_id, nodes in cn_communities.items():
        #             print(f"社区 {comm_id} (包含 {len(nodes)} 个文档)")
        #             print(f"  代表性文档: {', '.join(nodes[:3])}")
        #             if comm_id in cn_keywords:
        #                 print(f"  关键词: {', '.join([word for word, _ in cn_keywords[comm_id][:10]])}")
        #             print()
    else:
        print("中文评论数据不足，跳过分析")
    
    # 处理英文评论
    if len(en_df) >= 2:
        print("\n===== 分析英文评论 =====")
        en_cosine_df, en_processed_df, en_tfidf_matrix = calculate_cosine_similarity(en_df)
        
        # 找出最相似的文档对
        en_similar_pairs = find_most_similar_pairs(en_cosine_df, threshold)
        print(f"\n英文评论中最相似的 {len(en_similar_pairs)} 个文档对:")
        for file1, file2, sim in en_similar_pairs:
            print(f"{file1} 和 {file2}: {sim:.4f}")
        
        # 保存结果
        en_output_file = os.path.join(output_dir, "english_cosine_similarity.csv")
        en_cosine_df.to_csv(en_output_file)
        print(f"英文评论相似度矩阵已保存到 {en_output_file}")
        
        # 保存详细的相似对结果
        en_pairs_output = os.path.join(output_dir, "english_similar_pairs.csv")
        with open(en_pairs_output, 'w', encoding='utf-8') as f:
            f.write("File1,File2,Similarity\n")
            for file1, file2, sim in en_similar_pairs:
                f.write(f"{file1},{file2},{sim:.6f}\n")
        print(f"英文评论最相似文档对已保存到 {en_pairs_output}")
        
        # 可视化
        print("\n生成英文评论可视化...")
        
        # # 网络图
        # en_graph = plot_similarity_network(en_cosine_df, "英文评论", threshold,
        #                                 os.path.join(output_dir, "english_network.png"))
        
        # 降维可视化
        plot_dimensionality_reduction(en_tfidf_matrix, en_df, "英文评论", 
                                    output_file=os.path.join(output_dir, "english_pca.png"))
        
        
        # # 分析聚类
        # if en_graph:
        #     en_communities, en_keywords = analyze_clusters(en_graph, en_df)
        #     if en_communities:
        #         print("\n英文评论中的社区分析:")
        #         for comm_id, nodes in en_communities.items():
        #             print(f"社区 {comm_id} (包含 {len(nodes)} 个文档)")
        #             print(f"  代表性文档: {', '.join(nodes[:3])}")
        #             if comm_id in en_keywords:
        #                 print(f"  关键词: {', '.join([word for word, _ in en_keywords[comm_id][:10]])}")
        #             print()
    else:
        print("英文评论数据不足，跳过分析")
    
    print("\n分析完成！所有结果已保存到", output_dir)

if __name__ == "__main__":
    main()