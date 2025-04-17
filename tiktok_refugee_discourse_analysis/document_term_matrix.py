# Document-term matrix exploration with Spacy for lemmatization
import os
import json
import pandas as pd
import jieba
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import defaultdict

# 加载spacy模型 - 需要事先安装
# pip install spacy
# python -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")
print("Spacy模型加载成功")

# 修改停用词加载函数，使其一次性处理中英文停用词
def load_stopwords(cn_stopwords_path, en_stopwords_path=None):
    """加载停用词"""
    try:
        # 加载中文停用词
        cn_stopwords = set(open(cn_stopwords_path, encoding="utf-8").read().splitlines())
    except FileNotFoundError:
        print(f"警告: 未找到中文停用词文件 '{cn_stopwords_path}'，将使用空集合")
        cn_stopwords = set()
    
    # 英文停用词加载
    try:
        # 如果提供了英文停用词文件，则从文件加载
        if en_stopwords_path:
            en_stopwords = set(open(en_stopwords_path, encoding="utf-8").read().splitlines())
        else:
            # 否则使用Spacy的默认停用词
            en_stopwords = set(nlp.Defaults.stop_words)
    except FileNotFoundError:
        print(f"警告: 未找到英文停用词文件 '{en_stopwords_path}'，将使用Spacy默认停用词")
        en_stopwords = set(nlp.Defaults.stop_words)
    
    print(f"加载了 {len(en_stopwords)} 个英文停用词和 {len(cn_stopwords)} 个中文停用词")
    
    return cn_stopwords, en_stopwords

def preprocess_text(text, cn_stopwords, en_stopwords):
    """使用Spacy进行文本预处理和词形还原"""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 统一文本格式
    text = text.lower()  # 统一英文为小写
    text = re.sub(r'[^\w\s]', '', text)  # 去除标点符号
    text = re.sub(r'\d+', '', text)  # 去除数字

    # 分词和词形还原
    words = []
    
    # 中文分词 (中文不进行词形还原)
    cn_text = ""
    en_text = ""
    
    # 简单分离中英文
    for char in text:
        if ord(char) > 127:  # 中文字符
            cn_text += char
        else:  # 英文字符
            en_text += char
    
    # 处理中文
    for word in jieba.lcut(cn_text):
        if word.strip() and word not in cn_stopwords:
            words.append(word)
    
    # 处理英文 (使用Spacy进行词形还原)
    if en_text.strip():
        doc = nlp(en_text)
        for token in doc:
            # 获取词形还原结果
            lemma = token.lemma_ if token.lemma_ != "-PRON-" else token.text
            
            # 检查原词和词形还原结果是否为停用词
            if token.is_stop or lemma in en_stopwords or token.text in en_stopwords:
                continue
                
            if (token.text.isalpha() and len(token.text) > 1):
                words.append(lemma)

    return " ".join(words)  # 重新组合成字符串

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

def process_json_folder(folder_path, cn_stopwords, en_stopwords, category_field=None):
    """处理JSON文件夹，提取文本内容"""
    all_data = []
    fields_to_process = ['desc', 'comments_CN', 'comments_NonCN']
    # fields_to_process = ['comments_CN']
    
    # 遍历文件夹中的所有JSON文件
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                try:
                    # 读取JSON文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 提取各字段内容
                    document_texts = {}
                    
                    # 获取类别（如果指定了类别字段）
                    category = None
                    if category_field and category_field in data:
                        category = data[category_field]
                    
                    # 处理各个字段
                    for field in fields_to_process:
                        if field in data:
                            if field == 'desc':
                                # 描述字段通常是字符串
                                text = data[field] if isinstance(data[field], str) else ""
                            else:
                                # 评论字段可能是列表或字符串
                                text = extract_text_from_comments(data[field])
                            
                            # 预处理文本
                            processed_text = preprocess_text(text, cn_stopwords, en_stopwords)
                            document_texts[field] = processed_text
                    
                    # 添加到数据列表
                    all_data.append({
                        'file': file,
                        'category': category,
                        **document_texts
                    })
                    
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    # 将数据转换为DataFrame
    df = pd.DataFrame(all_data)
    
    return df

def create_tfidf_matrix(df, text_column, category_column=None, min_df=2, max_df=0.7, ngram_range=(1,1)):
    """创建TF-IDF矩阵，获取关键词"""
    # 创建 TF-IDF 向量化器，支持n-gram
    tfidf_vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    
    # 如果没有指定类别，或者类别列不存在，则计算整体TF-IDF
    if category_column is None or category_column not in df.columns:
        category_keywords = {}
        
        # 确保文本列存在且有数据
        if text_column in df.columns and not df[text_column].isna().all():
            # 计算TF-IDF
            tfidf_matrix = tfidf_vectorizer.fit_transform(df[text_column].fillna(''))
            feature_names = tfidf_vectorizer.get_feature_names_out()
            
            # 计算所有文档的平均TF-IDF值
            word_freq = dict(zip(feature_names, tfidf_matrix.mean(axis=0).A1))
            
            # 选出高频词
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:30]
            category_keywords['all'] = sorted_words
    else:
        # 按类别计算TF-IDF
        category_keywords = {}
        
        for category in df[category_column].dropna().unique():
            # 取出该类别的所有文档
            category_docs = df[df[category_column] == category][text_column].fillna('')
            
            if len(category_docs) > 0:
                # 计算TF-IDF
                tfidf_matrix = tfidf_vectorizer.fit_transform(category_docs)
                feature_names = tfidf_vectorizer.get_feature_names_out()
                
                # 计算词频
                word_freq = dict(zip(feature_names, tfidf_matrix.sum(axis=0).A1))
                
                # 选出高频词
                sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:30]
                category_keywords[category] = sorted_words
    
    return category_keywords

def visualize_keywords(category_keywords, title_prefix="", font_path=None):
    """可视化关键词（词云和柱状图）"""
    for category, words in category_keywords.items():
        if not words:
            continue
            
        # 词云部分
        try:
            # 构建词云参数
            wordcloud_params = {
                'width': 800, 
                'height': 400, 
                'background_color': "white",
                'prefer_horizontal': 1.0  # 尽量水平显示词语
            }
            
            # 安全检查字体文件
            if font_path and os.path.exists(font_path):
                wordcloud_params['font_path'] = font_path
                print(f"使用字体: {font_path}")
            else:
                print(f"警告: 找不到字体文件 '{font_path}'，尝试使用系统可用字体")
                
                # 尝试使用常见的系统字体
                system_fonts = [
                    '/System/Library/Fonts/PingFang.ttc',  # macOS
                    '/System/Library/Fonts/STHeiti Light.ttc',  # macOS
                    '/System/Library/Fonts/Hiragino Sans GB.ttc',  # macOS
                    '/Library/Fonts/Arial Unicode.ttf',  # macOS
                    'C:/Windows/Fonts/msyh.ttf',  # Windows
                    'C:/Windows/Fonts/simhei.ttf',  # Windows
                    '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # Linux
                ]
                
                for system_font in system_fonts:
                    if os.path.exists(system_font):
                        wordcloud_params['font_path'] = system_font
                        print(f"自动使用系统字体: {system_font}")
                        break
            
            # 尝试生成词云
            wordcloud = WordCloud(**wordcloud_params).generate_from_frequencies(dict(words))
            
            # 创建图表
            plt.figure(figsize=(10, 6))
            
            # 词云图
            plt.subplot(1, 2, 1)
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"{title_prefix} - {category}")
            
            # 柱状图（前10个词）
            plt.subplot(1, 2, 2)
            top_words = words[:20]
            
            # 解决中文显示问题
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            plt.barh([word[0] for word in top_words][::-1], [word[1] for word in top_words][::-1])
            plt.title(f"Top 10 Keywords - {category}")
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"生成词云时出错: {str(e)}")
            print("仅生成柱状图...")
            
            # 如果词云失败，只生成柱状图
            plt.figure(figsize=(8, 6))
            top_words = words[:15]
            
            # 解决中文显示问题
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 使用英文标签作为后备
            labels = []
            for word in top_words[::-1]:
                try:
                    # 尝试以utf-8显示
                    label = word[0]
                    # 如果是纯中文且不能正确显示，添加序号
                    if all(ord(c) > 127 for c in label):
                        label = f"{label} ({len(labels)+1})"
                except:
                    label = f"Word {len(labels)+1}"
                labels.append(label)
            
            plt.barh(labels, [word[1] for word in top_words][::-1])
            plt.title(f"Top 15 Keywords - {title_prefix} - {category}")
            plt.tight_layout()
            plt.show()
            
            # 打印关键词到控制台，以防图表显示不正确
            print(f"\n{title_prefix} - {category} 的前15个关键词:")
            for i, (word, score) in enumerate(top_words, 1):
                print(f"{i}. {word}: {score:.4f}")

def main():
    # 设置
    json_folder_path = input("请输入JSON文件所在的文件夹路径: ")
    cn_stopwords_path = input("请输入中文停用词文件路径 (按Enter使用默认 'chinese_stopwords.txt'): ") or "chinese_stopwords.txt"
    en_stopwords_path = input("请输入英文停用词文件路径 (按Enter使用默认 'spacy_stopwords.txt'): ") or "spacy_stopwords.txt"
    # 自动检测系统上可用的中文字体
    available_fonts = []
    system_fonts = [
        '/System/Library/Fonts/PingFang.ttc',  # macOS
        '/System/Library/Fonts/STHeiti Light.ttc',  # macOS
        '/System/Library/Fonts/Hiragino Sans GB.ttc',  # macOS
        '/Library/Fonts/Arial Unicode.ttf',  # macOS
        'C:/Windows/Fonts/msyh.ttf',  # Windows
        'C:/Windows/Fonts/simhei.ttf',  # Windows
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # Linux
    ]
    
    print("\n检测系统上可用的中文字体:")
    for i, font_path in enumerate(system_fonts, 1):
        if os.path.exists(font_path):
            available_fonts.append(font_path)
            print(f"{i}. 找到字体: {font_path}")
    
    if not available_fonts:
        print("未找到系统中文字体，可能会影响中文显示")
        font_path = input("请输入中文字体完整路径 (可选): ")
    else:
        print("\n字体选项:")
        for i, font in enumerate(available_fonts, 1):
            print(f"{i}. {os.path.basename(font)}")
        print(f"{len(available_fonts)+1}. 自定义字体路径")
        
        font_choice = input(f"请选择字体 (1-{len(available_fonts)+1}，默认为1): ") or "1"
        
        try:
            choice_idx = int(font_choice) - 1
            if 0 <= choice_idx < len(available_fonts):
                font_path = available_fonts[choice_idx]
            else:
                font_path = input("请输入中文字体完整路径: ")
        except ValueError:
            print("输入无效，使用第一个可用字体")
            font_path = available_fonts[0] if available_fonts else None
    
    # 加载停用词
    cn_stopwords, en_stopwords = load_stopwords(cn_stopwords_path, en_stopwords_path)

    
    # TF-IDF参数设置
    print("\nTF-IDF和N-gram参数设置:")
    print("min_df: 最小文档频率 (1-5，默认为2)")
    print("max_df: 最大文档频率 (0.5-0.9，默认为0.7)")
    print("n-gram范围: 词组长度范围，格式为'最小值,最大值'")
    
    try:
        min_df = int(input("请输入最小文档频率 (按Enter使用默认值): ") or "2")
        max_df = float(input("请输入最大文档频率 (按Enter使用默认值): ") or "0.7")
        ngram_input = input("请输入n-gram范围 (如'1,2'表示单词和双词组合，按Enter使用默认值'1,1'): ") or "1,1"
        ngram_range = tuple(map(int, ngram_input.split(',')))
    except ValueError:
        print("输入无效，使用默认值 min_df=2, max_df=0.7, ngram_range=(1,1)")
        min_df = 2
        max_df = 0.7
        ngram_range = (1, 1)
    
    # 处理JSON文件
    print("正在处理JSON文件...")
    df = process_json_folder(json_folder_path, cn_stopwords, en_stopwords)
    
    # 显示数据基本信息
    print(f"共处理 {len(df)} 个JSON文件")
    
    # 处理和分析各个字段
    for field in ['desc', 'comments_CN', 'comments_NonCN']:
        if field in df.columns:
            print(f"\n分析字段: {field}")
            
            # 创建TF-IDF矩阵
            print("计算TF-IDF...")
            category_keywords = create_tfidf_matrix(df, field, min_df=min_df, max_df=max_df, ngram_range=ngram_range)
            
            # 可视化
            print("生成词云和关键词图表...")
            visualize_keywords(category_keywords, title_prefix=field, font_path=font_path)
            
            # 显示关键词
            print(f"\n{field}字段的关键词:")
            for category, words in category_keywords.items():
                print(f"  {category}: {', '.join([word[0] for word in words[:20]])}")

if __name__ == "__main__":
    main()