with open('some.txt') as file_, open('immune_split.csv', 'w') as csvfile:
    lines = [x for x in file_.read().strip().split('@') if x]
    writer = csv.writer(csvfile, delimiter='|')
    writer.writerow(('ID', 'Text'))
    for idx, line in enumerate(lines, 1):
        writer.writerow((idx, line.strip('@')))