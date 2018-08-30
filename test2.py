import sys
import os
from stat import * # ST_SIZE etc

files = []
#print(sys.argv)

#for folderName, subfolders, filenames in os.walk('testdata'):
for folderName, subfolders, filenames in os.walk(sys.argv[1]):
  #print('The current folder is ' + folderName)
  #for subfolder in subfolders:
  #  print('SUBFOLDER OF ' + folderName + ': ' + subfolder)
  for filename in filenames:
    path = folderName + '/' + filename
    st = os.stat(path)
    #print([st[ST_SIZE] , path])
    files.append([st[ST_SIZE], path])

#print(files)  
files.sort(key=lambda tup: tup[0], reverse=True)

cursize=-1
samesizefiles = []
for size, name in files:
  print(size, name)  
  if size == cursize:
    print('same size')
    samesizefiles.append(name)
  else:
    cursize = size
    samesizefiles = [name]
  print(samesizefiles)
