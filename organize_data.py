# data from https://www.familyeducation.com/baby-names/browse-origin/surname

def get_names(filename):
  with open(filename, 'r') as ins:
    for line in ins:
      raw_str = line

  names_not_unique = [x.strip() for x in raw_str.split(',')]
  names = []
  for name in names_not_unique:
    if name not in names and len(name) > 0:
      names.append(name)
  return names

chinese_names = get_names('data/chinese_names.txt')
japanese_names = get_names('data/japanese_names.txt')
vietnamese_names = get_names('data/vietnamese_names.txt')
korean_names = get_names('data/korean_names.txt')

csv_str = ''
for name in chinese_names:
  csv_str += name.lower()+',chinese\n'
for name in japanese_names:
  csv_str += name.lower()+',japanese\n'
for name in vietnamese_names:
  csv_str += name.lower()+',vietnamese\n'
for name in korean_names:
  csv_str += name.lower()+',korean\n'

csv_file = open('names.csv', 'w')
csv_file.write(csv_str)
csv_file.close()