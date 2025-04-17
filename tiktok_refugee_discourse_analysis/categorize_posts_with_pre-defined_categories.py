import json
from zhipuai import ZhipuAI
from openai import OpenAI
from tqdm import tqdm
import os

# Load JSON file with posts
input_folder = "/Users/oliviafeng/Desktop/uchi/digitaltext1/final_project/divided_by_category/Uncategorized"
output_folder = "/Users/oliviafeng/Desktop/uchi/digitaltext1/final_project/divided_by_category/Uncategorized_done"
archive_folder = "/Users/oliviafeng/Desktop/uchi/digitaltext1/final_project/divided_by_category/archive_folder"

# 确保输出和归档文件夹存在
os.makedirs(output_folder, exist_ok=True)
os.makedirs(archive_folder, exist_ok=True)

# 获取文件夹中的所有JSON文件
import glob
json_files = glob.glob(os.path.join(input_folder, "*.json"))
print(f"找到 {len(json_files)} 个JSON文件")

# 获取已处理的文件名（检查归档文件夹中的文件）
processed_files = set(os.path.basename(f) for f in glob.glob(os.path.join(archive_folder, "*.json")))
print(f"已处理文件数: {len(processed_files)}")

# 创建CSV记录文件
import csv
csv_file_path = os.path.join(output_folder, "categorization_results.csv")
if not os.path.exists(csv_file_path):
    with open(csv_file_path, mode="w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["file_name", "note_id", "title", "category"])

# Define system message with task instructions
SYSTEM_PROMPT = """
You are a cross-cultural communication researcher analyzing social media posts on RedNote related to the recent migration of foreign TikTok users to the RedNote platform. Your task is to categorize each post into only one most relevant categories based on its post content.

### Categories:

1. Platform Migration & Adaptation
1a. About TikTok Ban / Censorship (e.g., posts discussing TikTok restrictions, bans, algorithm bias)
1b. About Adjusting to RedNote (e.g., comparisons between TikTok and RedNote, migration struggles)

2. Cross-Cultural Communication & Identity
2a. Music Sharing & Cultural Exchange (e.g., sharing songs from different cultures, music traditions)
2b. National Identity & Stereotypes (e.g., "How do Americans see China?" or cultural comparisons)
2c. Food & Lifestyle & Fashion comparison and sharing(e.g., cultural curiosity about foreign foods, daily life differences)
2d. Fan Culture & Celebrity Sharing (e.g., discussions about artists/celebrities from different cultures)


3. Political Discourse
3a. Political & Social Movements (e.g., BLM, climate change, Meta boycotts)
3b. Free Speech & Digital Rights (e.g., debates on censorship, privacy, surveillance)
3c. Digital Nationalism & Globalization (e.g., debates on Chinese vs. Western digital spaces)
3d. International Relations & Geopolitics

4. Marketing & E-Commerce
4a. Advertising Strategies
4b. Consumer Behavior & Trends
4c. Platform Monetization Debates

5. Creative Expression 
5a. Memes & Humor (e.g., internet humor, meme adaptation across platforms)
5b. Creative Hobbies & DIY & talent show (e.g., crafts, handmade items, creative projects, displaying personal artistic skills)

6. Social Interaction & Personal Narratives
6a. Self-Introduction & Personal Stories (e.g., pets, self-introduction posts, emotional reflections, daily life)
6b. Friendship & Community Building (e.g., posts seeking new friends on RedNote)
6c. Relationship Advice & Experiences
6d. Personal Growth & Life Reflections

7. Unknown / Not Relevant (If the post does not fit any of the above categories, you can mark it 7 and then name a new category, like "7: Other")
"""

# Function to generate user prompt with post data
def generate_user_prompt(post):
    # 处理comments字段，确保它是字符串
    comments = post.get('comments', '')
    if isinstance(comments, list):
        comments = "\n".join(comments) if comments else ""
    
    return f"""
### Post to Categorize:
Title: {post.get('title', 'No Title')}
Description: {post.get('desc', 'No Description')}
Comments: {comments}
Tags: {post.get('tag_list', '')}
Engagement: {post.get('liked_count', 'Unknown')} likes

Return only one category labels, e.g., "1a".
"""

# Function to parse AI response
def parse_ai_response(response_text):
    try:
        response_text = response_text.strip()
        print(f"🔹 Raw AI Response: {response_text}")  # Debugging print

        # 尝试提取单个数字类别
        import re
        # category_match = re.search(r'[1-7]', response_text)
        category_match = re.search(r'([1-7][a-e]?)', response_text)
        if category_match:
            return {"category": category_match.group(0)}
        else:
            print("⚠️ No categories extracted.")
            return {"category": "error"}

    except Exception as e:
        print(f"❌ Error parsing AI response: {e}")
        return {"category": "error"}

# 批处理功能
def process_batch(client, model, batch_files, batch_size=10):
    csv_records = []
    processed_batch = []
    
    for file_path in tqdm(batch_files, desc=f"Processing batch of {len(batch_files)} files"):
        try:
            file_name = os.path.basename(file_path)
            
            # 检查是否已处理
            if file_name in processed_files:
                continue
            
            # 加载JSON文件
            with open(file_path, "r", encoding="utf-8") as f:
                post = json.load(f)
            
            # 获取帖子ID
            note_id = post.get('note_id', '')
            title = post.get('title', '')
            
            print(f"Processing post: {note_id} - {title[:30]}...")
            user_prompt = generate_user_prompt(post)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]

            response = client.chat.completions.create(
                model=model,
                messages=messages
            )

            assistant_response = response.choices[0].message.content
            
            # 解析响应
            response_json = parse_ai_response(assistant_response)
            category = response_json.get("category", "error")
            
            # 更新帖子数据
            post["category"] = category
            
            # 保存到输出文件夹
            output_file_path = os.path.join(output_folder, file_name)
            with open(output_file_path, "w", encoding="utf-8") as f:
                json.dump(post, f, ensure_ascii=False, indent=4)
            
            # 添加到CSV记录
            csv_records.append([file_name, note_id, title, category])
            
            # 添加到已处理批次
            processed_batch.append((file_path, archive_folder))
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    
    # 更新CSV文件
    with open(csv_file_path, mode="a", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        for record in csv_records:
            writer.writerow(record)
    
    # 移动已处理文件
    for source_file, dest_folder in processed_batch:
        file_name = os.path.basename(source_file)
        dest_path = os.path.join(dest_folder, file_name)
        try:
            import shutil
            shutil.copy2(source_file, dest_path)  # 复制到归档文件夹
            os.remove(source_file)  # 删除原始文件
            processed_files.add(file_name)  # 添加到已处理集合
        except Exception as e:
            print(f"文件移动失败 {file_name}: {e}")
    
    return len(processed_batch)

# 主处理函数
def call_ai(client, model, max_files=20000, batch_size=10):
    # 过滤掉已处理的文件
    files_to_process = [f for f in json_files if os.path.basename(f) not in processed_files]
    
    if max_files:
        files_to_process = files_to_process[:max_files]
    
    print(f"需要处理 {len(files_to_process)} 个文件")
    
    # 分批处理
    total_processed = 0
    for i in range(0, len(files_to_process), batch_size):
        batch = files_to_process[i:i+batch_size]
        processed = process_batch(client, model, batch, batch_size)
        total_processed += processed
        print(f"已处理批次 {i//batch_size + 1}，总共处理 {total_processed} 个文件")
    
    print(f"✅ 处理完成，共处理 {total_processed} 个文件，结果保存到 {output_folder}")
    
    # 生成统计数据
    try:
        # 读取结果CSV
        category_counts = {}
        with open(csv_file_path, "r", encoding="utf-8", newline="") as file:
            reader = csv.reader(file)
            next(reader)  # 跳过标题行
            for row in reader:
                if len(row) >= 4:
                    category = row[3]
                    if category in category_counts:
                        category_counts[category] += 1
                    else:
                        category_counts[category] = 1
        
        # 保存统计结果
        with open(os.path.join(output_folder, "category_summary.json"), "w", encoding="utf-8") as f:
            json.dump({
                "total_processed": total_processed,
                "category_distribution": category_counts
            }, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"统计信息生成失败: {e}")

# 使用智谱AI进行分类
client = ZhipuAI(api_key="72d38a74be1e3bb61c552a7e105cfce3.ix1koYYSjZO9w7f4")
call_ai(client, "glm-4-flash", max_files=20000, batch_size=10)

# 如果要测试其他模型，取消相应的注释
"""
# deepseek-chat
client = OpenAI(
    api_key="sk-32c365525bef4c6a9f20dfeaa6dc1f8e",
    base_url="https://api.deepseek.com",
)
call_ai(client, "deepseek-chat", max_files=20000, batch_size=10)
"""