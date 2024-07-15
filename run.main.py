import os
files_to_remove = [
    'processed_all_words_data.csv',
    'processed_text_files.csv',
    'processed_tfidf_data.csv',
    'text_files.csv',
    'word_vectors.csv'
]

# Remove files if they exist
for file_name in files_to_remove:
    if os.path.exists(file_name):
        os.remove(file_name)
        print(f"Removed {file_name}")
    else:
        print(f"{file_name} not found, skipping removal")
os.system('python installmodules.py && python pdf_ocr.py && python txt_to_csv_whole.py && python process-csv.py && python tf-idf.py && python t.words.py && python t.t.vecors.bert_copy.py')