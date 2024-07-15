import pandas as pd
import os
import numpy as np
import warnings
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import concurrent.futures
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from multiprocessing import Pool, cpu_count

warnings.filterwarnings("ignore", message="`resume_download` is deprecated")

# 加载NLTK的停用词和分词器
#nltk.download('stopwords')
#nltk.download('punkt')

# 加载包含词汇和词向量的CSV文件
csv_path = 'word_vectors.csv'

# 提取词汇和向量
print("Loading CSV file to RAM...")
# 使用tqdm读取CSV文件
df = pd.read_csv(csv_path)
selected_words = df['Words'].tolist()
word_vectors = np.array([eval(vec) for vec in tqdm(df['Vectors'], desc="Loading word vectors")])  # 使用tqdm显示进度


# 加载英语停用词
stop_words = set(stopwords.words('english'))

# 加载预训练的BERT模型
print("Loading model and auto config...")
model = SentenceTransformer('bert-base-nli-mean-tokens')

def extract_keywords_tfidf(text):
    # 使用正则表达式去除非字母数字字符和单个字母
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\b\w\b', '', text)
    
    # 分词并去除停用词
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    
    # 初始化TF-IDF向量化器
    tfidf_vectorizer = TfidfVectorizer()
    
    # 对文本进行TF-IDF向量化
    tfidf_matrix = tfidf_vectorizer.fit_transform([filtered_text])
    
    # 获取特征名（关键词）和对应的TF-IDF值
    feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_values = tfidf_matrix.toarray()[0]
    
    # 将关键词和TF-IDF值一一对应
    keyword_tfidf = {feature_names[i]: tfidf_values[i] for i in range(len(feature_names))}
    
    return keyword_tfidf

def find_similar_words(query_text, top_n=10):
    # 使用BERT模型向量化查询文本中的关键词
    query_vector = model.encode([query_text])[0].astype(np.float32)  # 显式转换为float32
    
    # 计算查询向量与词汇表中每个词向量的余弦相似度
    similarities = util.pytorch_cos_sim(query_vector, word_vectors.astype(np.float32))[0].tolist()  # 显式转换为float32
    
    # 获取相似度最高的词汇
    similar_indices = np.argsort(similarities)[-top_n:][::-1]
    similar_words = [(selected_words[i], similarities[i]) for i in similar_indices]
    
    return similar_words

def find_similar_words_parallel(query_text, top_n=3):
    # 提取关键词及其TF-IDF值
    keyword_tfidf = extract_keywords_tfidf(query_text)
    keywords = list(keyword_tfidf.keys())
    
    # 使用BERT模型向量化关键词
    keyword_vectors = model.encode(keywords).astype(np.float32)  # 显式转换为float32
    
    # 并行查询每个关键词的相似词汇
    with concurrent.futures.ThreadPoolExecutor() as executor:
        keyword_results = []
        # 使用tqdm显示并行进度
        with tqdm(total=len(keywords), desc=f'Finding similar words for query "{query_text}"') as pbar:
            futures = {executor.submit(find_similar_words, keyword, top_n): keyword for keyword in keywords}
            for future in concurrent.futures.as_completed(futures):
                keyword = futures[future]
                try:
                    result = future.result()
                    keyword_results.append((keyword, result, keyword_tfidf[keyword]))
                except Exception as e:
                    print(f"Exception occurred for keyword '{keyword}': {str(e)}")
                pbar.update(1)  # 更新tqdm进度条
    
    return keyword_results

# 结果保存列表
all_results = []

# 主循环
while True:
    # 提示用户输入问题
    user_question = input("please enter a question(prefer keywords or simple questions)\neg:What is chemical balance?:\n")
    
    # 检查输入是否为空或格式不正确
    if not user_question.strip():  # 如果输入为空
        print("{blank}")
        continue
    
    # 提取问题中的关键词，并使用BERT模型找到最相似的词汇
    try:
        print("\nExtracting keywords and calculating TF-IDF values...")
        keyword_results = find_similar_words_parallel(user_question)
    except Exception as e:
        print(f"{str(e)}")
        continue
    
    # 打印每个关键词及其相似词汇的结果，并保存到列表中
    print("\nResults:")
    for keyword, similar_words, tfidf_value in keyword_results:
        print(f"\nKeyword '{keyword}' (TF-IDF: {tfidf_value:.4f})")
        print("Similar words:")
        for word, similarity in similar_words:
            print(f"{word}: {similarity:.4f}")
            all_results.append([user_question, keyword, tfidf_value, word, similarity])
    # 将结果保存到CSV文件
    results_df = pd.DataFrame(all_results, columns=['Question', 'Keyword', 'TF-IDF', 'Similar Word', 'Similarity'])
    if os.path.exists('results.csv'):
        os.remove('results.csv')
        print("Removed 'results.csv'")
    else:
        print("'results.csv' not found, skipping removal")
    results_df.to_csv('results.csv', index=False)
    os.system('python t.results_in_all_words_t_speed_multy.py && python t.cal1.py')
    keyword_results = []
    keyword = []
    user_question = []
