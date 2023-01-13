import csv

with open('C:\\Users\\Predator\\Documents\\Document-Classification\\classifier\\immune.txt', 'r') as in_file:
    stripped = (line.strip() for line in in_file)
    lines = (line.split(",") for line in stripped if line)
    with open('C:\\Users\\Predator\\Documents\\Document-Classification\\classifier\\immune_row.csv', 'w') as out_file:
        writer = csv.writer(out_file)
        writer.writerow('text', ' ')
        writer.writerows(lines)