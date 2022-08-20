import os
from glob import glob
import numpy as np

def load_examples():
    examples_path = []
    folders_path = os.path.join('data', '*')
    folders_name = glob(folders_path)
    
    for folder in folders_name:
        files_path = os.path.join(folder, '*')
        files_name = glob(files_path)
        for i in range(50 // len(folders_name)):
            random_example = np.random.randint(0, len(files_name))
            examples_path.append(files_name[random_example])
            
    return examples_path

def examples():
    texts = []
    examples_path = load_examples()
    for path in examples_path:
        class_name = path.split(os.sep)[1]
        with open(path, 'r', encoding='utf8') as file:
            text = file.read()[:100]
            texts.append([text, class_name])
    return texts


if __name__ == '__main__':
    print(examples())