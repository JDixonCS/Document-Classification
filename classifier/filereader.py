# Import Module
import os

# Folder Path
path = "C:/Users/z3696/Documents/Document-Classification/classifier/NIST_TEXT/2010/neg"

# Change the directory
os.chdir(path)


# Read text File


def read_text_file(file_path):
    with open(file_path, 'r') as f:
        print(f.read())


# iterate through all file
for file in os.listdir():
    # Check whether file is in text format or not
    if file.endswith(".txt"):
        file_path = f"{path}/{file}"

        # call read text file function
        read_text_file(file_path)



import os

# folder path
dir_path = r'C:/Users/z3696/Documents/Document-Classification/classifier/TRAIN_SET'
count = 0
# Iterate directory
for path in os.listdir(dir_path):
    # check if current path is a file
    if os.path.isfile(os.path.join(dir_path, path)):
        count += 1
print('File count:', count)
'''
import os
import glob

os.chdir("C:/Users/z3696/Documents/Document-Classification/classifier/NIST_TEXT")
names = {}
for fn in glob.glob("**/*.txt"):
    with open(fn) as f:
        names[fn] = sum(1 for line in f if line.strip() and not line.startswith('#'))

print(names)

#!/usr/bin/env python3
'''
from pathlib import Path

target = Path("C:/Users/z3696/Documents/Document-Classification/classifier/TRAIN_SET")

names = {}

for file in target.glob("**/*.txt"):
    with file.open("rt") as f:
        names[f.name] = sum(
            1 for line in f
            if line.strip() and not line.startswith('#')
        )

print(names)

