import glob

read_files = glob.glob("C:\\Users\\z3696\\Documents\\Document-Classification\\classifier\\TEST_SET\\*.txt")

with open("FULL-TEST.txt", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())