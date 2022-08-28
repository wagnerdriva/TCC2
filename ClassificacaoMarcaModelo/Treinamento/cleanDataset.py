import os


dirName = '/home/users/wagner/TCC2/ClassificacaoMarcaModelo/Treinamento/Data/HATCH'

# Get the list of all files in directory tree at given path
listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(dirName):
    print(dirpath, dirnames, filenames)
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    

import PIL
from PIL import Image
from tqdm import tqdm

count = 0
# Print the files    
for elem in tqdm(listOfFiles):
    try:
        im = Image.open(elem)
        im.verify() #I perform also verify, don't know if he sees other types o defects
        im.close() #reopen is necessary in my case
        im = Image.open(elem) 
        im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        im.close()
    except:
        os.remove(elem)
        count += 1
        print(elem)
        print(f"Total de erros: {count}")
