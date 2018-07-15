"""
updates url.txt for the open data labeling tool on shadysource.github.io
"""
import os

try:
    os.remove('url.txt')
except:
    pass
dirs = os.listdir('dataset')
dirs.sort()

for i in dirs:
    if '.' in i:
        dirs.remove(i)

with open('url.txt', 'w') as f:
    for i in dirs:
        f.write(i + ' ')
    f.write('\n')

    for i in dirs:
        images = os.listdir(os.path.join('dataset',i))
        for name in images:
            if " " in name:
                os.rename(os.path.join('dataset',i, name), os.path.join('dataset',i, name.replace(" ", "_")))
            f.write('http://140.115.152.223:7001/data/dataset/'+i+'/'+name+' ')
        f.write('\n')
