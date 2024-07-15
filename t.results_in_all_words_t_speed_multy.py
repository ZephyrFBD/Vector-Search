import pandas as pd
from tqdm import tqdm
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed

# 读取results.csv文件
results_df = pd.read_csv('results.csv')

# 读取processed_all_words_data.csv文件
processed_df = pd.read_csv('processed_all_words_data.csv')

# 提取results.csv中的Similar Word列并转为集合
similar_words = set(results_df['Similar Word'].unique())

# 初始化总体进度条
total_progress = tqdm(total=len(similar_words), desc="Overall Progress")

matching_rows = []

# 预处理processed_df，创建TF-IDF字典以加速检索
tfidf_dict = {}
for idx, row in processed_df.iterrows():
    tfidf_dict[idx] = dict(ast.literal_eval(row['TF-IDF']))

# 定义处理函数
def process_word(word):
    word_matches = []
    for idx, row in processed_df.iterrows():
        if word in row['All Words']:
            matching_word_tfidf = tfidf_dict[idx].get(word, 0.0)
            if matching_word_tfidf != 0.0:
                keyword_row = results_df.loc[results_df['Similar Word'] == word].iloc[0]  # 获取关键词对应的行
                word_matches.append({
                    'Question': keyword_row['Question'],
                    'Keyword': keyword_row['Keyword'],
                    'Similarity': keyword_row['Similarity'],
                    'Keyword TF-IDF': keyword_row['TF-IDF'],
                    'Matching Word': word,
                    'Matching Word TF-IDF': matching_word_tfidf,
                    'File Text': row['File Text'],
                    'File Path': row['File Path']
                })
    return word_matches

# 使用线程池进行并行处理
with ThreadPoolExecutor(max_workers=None) as executor:  # None 表示使用系统最大线程数
    futures = [executor.submit(process_word, word) for word in similar_words]
    
    # 处理结果
    for future in as_completed(futures):
        matching_rows.extend(future.result())
        total_progress.update(1)

total_progress.close()

# 将匹配的行转换为DataFrame
matching_rows_df = pd.DataFrame(matching_rows)

# 将结果保存到CSV文件
output_df = matching_rows_df[['Question', 'Keyword', 'Similarity', 'Keyword TF-IDF', 'Matching Word', 'Matching Word TF-IDF', 'File Text', 'File Path']]
output_df.to_csv('matching_file_paths.csv', index=False)

print("结果已保存到 matching_file_paths.csv")
