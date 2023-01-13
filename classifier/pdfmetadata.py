from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument

fp = open('C:\\Users\\Predator\\Documents\\Document-Classification\\backend\\COVID19PDF\\71.full.pdf', 'rb')
parser = PDFParser(fp)
doc = PDFDocument(parser)

print(doc.info)  # The "Info" metadata