import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

# 加载 CSV 文件
csv_path = 'processed_text_files.csv'
df = pd.read_csv(csv_path)

# 提取文本数据
documents = df['File Text'].tolist()

# 初始化 TF-IDF 向量化器
tfidf_vectorizer = TfidfVectorizer()

# 计算 TF-IDF
tfidf_vectors = tfidf_vectorizer.fit_transform(documents)

# 将 TF-IDF 结果添加到 DataFrame 的第五列
df['TF-IDF'] = [""] * len(documents)
feature_names = tfidf_vectorizer.get_feature_names_out()

with tqdm(total=len(documents), desc='Processing rows', unit='row') as pbar:
    for i in range(len(documents)):
        tfidf_scores = list(zip(feature_names, tfidf_vectors[i].toarray()[0]))
        tfidf_scores_sorted = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)[:10]  # 只存储前10个最高的 TF-IDF 值
        df.at[i, 'TF-IDF'] = tfidf_scores_sorted
        pbar.update(1)

# 保存处理后的 DataFrame 到新的 CSV 文件
output_csv = './processed_tfidf_data.csv'
df.to_csv(output_csv, index=False)

print(f"Processed TF-IDF data saved to '{output_csv}'.")
