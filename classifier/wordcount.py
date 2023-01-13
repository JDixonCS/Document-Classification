file = open("C:\\Users\\Predator\\Documents\\Document-Classification\\classifier\\COVID19PDF\\Immune\\immune.txt", "rt")
data = file.read()
words = data.split()

print('Number of words in text file :', len(words))