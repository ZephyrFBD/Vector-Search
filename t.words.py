import pandas as pd
from tqdm import tqdm
import ast

# 加载 CSV 文件
csv_path = 'processed_tfidf_data.csv'
df = pd.read_csv(csv_path)

# 提取第五列的内容
tfidf_data = df['TF-IDF'].tolist()

# 创建一个新的列来存储所有词
df['All Words'] = [""] * len(tfidf_data)

# 处理每一行，并添加进度条显示
with tqdm(total=len(tfidf_data), desc='Processing rows', unit='row') as pbar:
    for i in range(len(tfidf_data)):
        # 将字符串转换回原来的列表格式
        tfidf_list = ast.literal_eval(tfidf_data[i])
        # 提取所有词
        words = [word for word, _ in tfidf_list]
        # 格式化为所需的格式："词, 词, 词, ..."
        formatted_words = ", ".join(words)
        df.at[i, 'All Words'] = formatted_words
        pbar.update(1)

# 保存处理后的 DataFrame 到新的 CSV 文件
output_csv = './processed_all_words_data.csv'
df.to_csv(output_csv, index=False)

print(f"Processed data saved to '{output_csv}'.")
