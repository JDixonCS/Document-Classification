import string
import glob
# replacement strings
WINDOWS_LINE_ENDING = b'\r\n'
UNIX_LINE_ENDING = b'\n'

read_files = glob.glob("C:\\Users\\Predator\\Documents\\Document-Classification\\backend\\COVID19PDF\\Transmission\\*.txt")
# relative or absolute file path, e.g.:
#file_path = r""

with open(read_files, 'rb') as open_file:
    for f in read_files:
        content = open_file.read()

content = content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)

with open(read_files, 'wb') as open_file:
    for f in read_files:
        open_file.write(content)