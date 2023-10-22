from os.path import isfile, join
from os import listdir
from wordaffect import EmotionDetector
from multiprocessing import Pool

import nltk

#progress bar
from tqdm import tqdm

def process_single_file(vars):
    inputfile, outputfile = vars
    print(inputfile)
    ed = EmotionDetector()
    ed.detect_emotion_in_file(inputfile, outputfile)



if __name__ == '__main__':
    #nltk.download("wordnet")
    #nltk.download('averaged_perceptron_tagger')

    #mypath = "/home/ssamot/projects/github/gutenberg/processed/texts"
    #mypath = "/scratch/ssamot/texts/"
    mypath = "./texts"
    #outpath = "/home/ssamot/projects/github/gutenberg/processed/results"
    outpath = "./results"
    onlyfiles = [ (join(mypath,f), f[:-4], f) for f in listdir(mypath) if isfile(join(mypath,f)) and f.endswith(".txt")]

    counts = {}
    input_output = []
    for files in onlyfiles:
        subject = files[1].split("_")[1]
        if(subject in counts):
            counts[subject]+= 1
        else:
            counts[subject] = 1
        input_output.append((files[0], join(outpath,files[2])))

    for key,val in counts.items():
        print(key, val)

    # may need to lower Number if "[Errno 12] Cannot allocate memory"
    # can increase if no error
    print("--------")
    p = Pool(10)
    length= len(input_output)
    print(length)
    print(input_output[0])
    print("--------")
    
    #map(process_single_file, input_output)
    for i,_ in enumerate(p.imap(process_single_file, input_output)):
        print("Progress:", i+1,"/",length,",", i+1/length)



