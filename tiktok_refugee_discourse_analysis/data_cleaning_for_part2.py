import os
import json
import re

def clean_json_desc(input_folder, output_folder):
    """
    遍历指定文件夹中的所有JSON文件，清理每个文件中的"desc"字段，
    移除格式为"#xxx[话题]#"的标签部分。
    
    Args:
        input_folder: 输入JSON文件的文件夹路径
        output_folder: 输出清理后JSON文件的文件夹路径
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 正则表达式模式
    # 匹配 #xxx[话题]# 格式
    pattern1 = r'#[^#]*\[话题\]#'
    # 匹配 [xxx] 或 [xxxX] 格式 (包括表情和其他方括号内容)
    pattern2 = r'\[[^\]]+\]'
    
    # 组合模式
    patterns = [pattern1, pattern2]
    
    # 要处理的字段列表
    fields_to_clean = ['desc', 'comments_CN', 'comments_NonCN']
    
    # 统计信息
    total_files = 0
    processed_files = 0
    
    # 遍历输入文件夹中的所有文件
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.json'):
                total_files += 1
                file_path = os.path.join(root, file)
                
                try:
                    # 读取JSON文件
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 处理所有需要清理的字段
                    for field in fields_to_clean:
                        if field in data and data[field]:
                            # 检查字段类型
                            if isinstance(data[field], str):
                                # 字符串类型直接处理
                                original_text = data[field]
                                cleaned_text = original_text
                                
                                # 应用所有清理模式
                                for pattern in patterns:
                                    cleaned_text = re.sub(pattern, '', cleaned_text)
                                
                                # 删除多余的空格
                                cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                                
                                # 更新数据
                                data[f'original_{field}'] = original_text  # 保留原始内容
                                data[field] = cleaned_text
                            elif isinstance(data[field], list):
                                # 列表类型需要逐个处理每个元素
                                original_list = data[field].copy()
                                cleaned_list = []
                                
                                for item in data[field]:
                                    if isinstance(item, str):
                                        cleaned_item = item
                                        # 应用所有清理模式
                                        for pattern in patterns:
                                            cleaned_item = re.sub(pattern, '', cleaned_item)
                                        cleaned_item = re.sub(r'\s+', ' ', cleaned_item).strip()
                                        cleaned_list.append(cleaned_item)
                                    elif isinstance(item, dict) and 'text' in item:
                                        # 如果是包含'text'键的字典
                                        item_copy = item.copy()
                                        if isinstance(item['text'], str):
                                            item_copy['original_text'] = item['text']
                                            cleaned_text = item['text']
                                            # 应用所有清理模式
                                            for pattern in patterns:
                                                cleaned_text = re.sub(pattern, '', cleaned_text)
                                            item_copy['text'] = re.sub(r'\s+', ' ', cleaned_text).strip()
                                        cleaned_list.append(item_copy)
                                    else:
                                        # 其他类型直接添加
                                        cleaned_list.append(item)
                                
                                # 更新数据
                                data[f'original_{field}'] = original_list
                                data[field] = cleaned_list
                    
                    # 确定输出文件路径 - 保持相同的目录结构
                    rel_path = os.path.relpath(root, input_folder)
                    output_dir = os.path.join(output_folder, rel_path)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    output_file = os.path.join(output_dir, file)
                    
                    # 写入清理后的数据
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    
                    processed_files += 1
                    
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")
    
    print(f"处理完成！共处理 {processed_files}/{total_files} 个JSON文件。")
    return processed_files, total_files

def main():
    # 设置输入和输出文件夹
    input_folder = input("请输入JSON文件所在的文件夹路径: ")
    output_folder = input("请输入清理后JSON文件的保存路径: ")
    
    if not os.path.exists(input_folder):
        print(f"错误：输入文件夹 '{input_folder}' 不存在！")
        return
    
    # 运行清理函数
    processed, total = clean_json_desc(input_folder, output_folder)
    
    if processed > 0:
        print(f"成功清理了 {processed} 个JSON文件，保存在 '{output_folder}' 文件夹中。")
    else:
        print("未找到需要处理的JSON文件。")

if __name__ == "__main__":
    main()