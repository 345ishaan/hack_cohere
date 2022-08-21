import os
import numpy as np
import easyocr as ocr
import string
import json

from PIL import Image
from glob import glob

def load_examples(dir_name):
    examples_path = []
    folders_path = os.path.join(dir_name, '*')
    folders_name = glob(folders_path)
    
    for folder in folders_name:
        files_path = os.path.join(folder, '*')
        files_name = glob(files_path)
        examples_path.extend(files_name[:5])
            
    return examples_path

def get_ocr_model():
    model = ocr.Reader(['en'],model_storage_directory='.')
    return model

def examples(dir_name, use_cache=False):
    texts = []
    examples_path = load_examples(dir_name)
    model = get_ocr_model()
    # load cache
    cache = {}
    if use_cache and os.path.isfile('cache.json'):
        with open('cache.json') as fp:
            cache = json.load(fp)

    for path in examples_path:
        class_name = path.split(os.sep)[1]
        if class_name in cache and path in cache[class_name]:
            if cache[class_name][path]:
                texts.append([cache[class_name][path], class_name])
        else:
            input_image = Image.open(path)
            result = model.readtext(np.array(input_image))
            valid_words = []
            for res in result:
                if res[2] > 0.5:
                    text = res[1].lower()
                    text = text.strip()
                    strip = ''.join(c for c in text if c.isalpha())
                    if len(strip) > 2:
                        valid_words.append(strip)
            concat_words = ' '.join(valid_words)
            if class_name not in cache:
                cache[class_name] = {}
            cache[class_name][path] = concat_words
            if concat_words:
                texts.append([concat_words, class_name])
    
    if use_cache:
        with open("cache.json", "w") as write_file:
            json.dump(cache, write_file)
    
    return texts


if __name__ == '__main__':
    outputs = (examples('doc_data', use_cache=True))
    for op in outputs:
        print(op[0], op[1])
