import os
import shutil
sourcepath='C:/Users/Predator/Documents/Document-Classification/backend/MLPDF'
sourcefiles = os.listdir(sourcepath)
destinationpath = 'C:/Users/Predator/Documents/Document-Classification/backend/XPDF-TXT'
for file in sourcefiles:
    if file.endswith('.txt'):
        shutil.move(os.path.join(sourcepath,file), os.path.join(destinationpath,file))