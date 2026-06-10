import os
import shutil
import nltk

nltk_data_path = os.path.join(os.getenv('APPDATA'), 'nltk_data')
corpora_path = os.path.join(nltk_data_path, 'corpora')
wordnet_zip = os.path.join(corpora_path, 'wordnet.zip')
wordnet_dir = os.path.join(corpora_path, 'wordnet')

if os.path.exists(wordnet_zip):
    os.remove(wordnet_zip)
if os.path.exists(wordnet_dir):
    shutil.rmtree(wordnet_dir)

nltk.download('wordnet', force=True)
