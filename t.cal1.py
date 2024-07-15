import pandas as pd
import os

# 读取 matching_file_paths.csv 文件
matching_df = pd.read_csv('matching_file_paths.csv')

# 查找具有相同 File Path 但有多个 Matching Word 的行
duplicate_file_paths = matching_df.groupby('File Path').filter(lambda x: len(x) > 1)

# 输出具有多个 Matching Word 的 File Path 和对应的 Matching Words，按 Matching Word TF-IDF 排序
print("具有多个 Matching Word 的 File Path 和对应的 Matching Words（按 Matching Word TF-IDF 排序）:")
for file_path, group in duplicate_file_paths.groupby('File Path'):
    # 按 Matching Word TF-IDF 排序
    sorted_group = group.sort_values(by='Matching Word TF-IDF', ascending=False)
    
    matching_words = sorted_group['Matching Word'].tolist()
    tfidf_values = sorted_group['Matching Word TF-IDF'].tolist()
    
    print(f"File Path: {file_path}")
    for word, tfidf in zip(matching_words, tfidf_values):
        print(f"Matching Word: {word} (TF-IDF: {tfidf})")
    print()  # 输出空行进行分隔
os.remove('results.csv')
os.remove('matching_file_paths.csv')
