import io
import unicodedata
import glob
# replacement strings
WINDOWS_LINE_ENDING = b'\r\n'
UNIX_LINE_ENDING = b'\n'

src_path = "C:\\Users\\Predator\\Documents\\Document-Classification\\classifier\\industry.txt"
dst_path = "C:\\Users\\Predator\\Documents\\Document-Classification\\classifier\\industry-ch.txt"

read_files = glob.glob("C:\\Users\\Predator\\Documents\\Document-Classification\\classifier\\text-data\\sigir-2010\\neg\\*.txt")
write_files = glob.glob("C:\\Users\\Predator\\Documents\\Document-Classification\\classifier\\text-win\\sigir-2010\\neg\\*.txt")
for f in read_files:
        with io.open(f, mode="r", encoding="utf8") as fd:
            content = fd.read()
            with io.open(f, mode="w", encoding="cp1252") as fd:
                fd.write(content)

with open(dst_path, 'rb') as open_file:
    content = open_file.read()

content = content.replace(UNIX_LINE_ENDING, WINDOWS_LINE_ENDING)

with open(dst_path, 'wb') as open_file:
    open_file.write(content)