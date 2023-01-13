# folder with 1000s of PDFs
dest <- "C:\\Users\\Predator\\Documents\\Document-Classification\\backend\\COVID19PDF\\Transmission"

# make a vector of PDF file names
myfiles <- list.files(path = dest, pattern = "pdf",  full.names = TRUE)
myfiles
# convert each PDF file that is named in the vector into a text file 
# text file is created in the same directory as the PDFs
# note that my pdftotext.exe is in a different location to yours
lapply(myfiles, function(i) system(paste('"C:/Users/Predator/Program Space/xpdf-tools-win-4.03/xpdf-tools-win-4.03/bin64/pdftotext.exe"', 
                                         paste0('"', i, '"')), wait = FALSE) )
