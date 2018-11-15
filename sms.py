# coding: utf-8

#Data Science imports
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#Text Processing imports
import csv
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SpanishStemmer
import unicodedata
import re

#Absolute paths to files
# change as required
datafile='./SMS_DATA.csv'

# AUX function for PRE-PROCESSING sentences
def savetocsv(X, filename):
    data = pd.DataFrame(data=X)
    data.to_csv(filename)

#Create dicitonary of word lemmas
def load_lemas(filename='Lemas.txt'):
    lemas=pd.read_table(filename, sep=' ', header=None, engine='python')
    lemas.drop([0,2,3],axis=1,inplace=True)
    lemas.rename({1:'Palabra',4:'Lema'},axis=1,inplace=True)
    lemas['Lema']=lemas['Lema'].apply(lambda x: x.lower() if type(x)==str else x)
    locs=lemas.groupby('Lema').groups['-']
    lemas.drop(labels=locs,inplace=True)
    return lemas,locs

def lematizador(types,lem='GRAMPAL',strt_idx=0):
    #types=['des','íntimo','intenso','muchísimo','intensísimo','dieron','corriendo','muy','bien','diariamente']
    locs=[]
    if lem == 'GRAMPAL':
        lemas = lema_grampal(types,strt_idx)
    elif lem == 'GEDIC': 
        lemas = lema_gedic(types,strt_idx)
    elif lem == 'FILE':
        lemas,locs=load_lemas()
        print(f'Se cargó el archivo en lemas.DataFrame con {lemas.shape[0]} lemas')
    else:
        print(f'Lematizador {lem} NO DISPONIBLE')
    return lemas,locs


def preprocess_sentence(sentence,stemming=False):
    remove_table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('spanish'))

    tokens = word_tokenize(sentence)      # Tokenize
    tokens = [w.lower() for w in tokens]  # Lowercase
    stripped = [word.translate(remove_table) for word in tokens]     # Remove punctuation
    stripped = [word for word in stripped if word.isalpha()]         # Remove numerals
    stripped = [word for word in stripped if not word in stop_words] # Remove stopwords
    if stemming:
        stemmer = SpanishStemmer()
        stripped = [stemmer.stem(word) for word in stripped]               # Stemming
    return " ".join(stripped)

def preprocess_data(data,lang='spanish',txt_key='Texto',minSize=6,stpw=True,lematize=False,lem='GRAMPAL',stemming=False,strt_idx=0):
    remove_table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words(lang))
    
    # strip data text (word tokens)
    try:
        filtro = data[txt_key].str.split().str.len()>=minSize                       # Filter of minSize sentences
    except KeyError:
        print(f'Key Error exception: {txt_key}')
        return None, None
    data = data[filtro]                                                         # Keep sentences >= minSize
    # Lower case all, remove punctuation, remove numerals
    stripped = data[txt_key].str.split().apply(lambda x : [y.lower() for y in x]).\
                                        apply(lambda x : [y.translate(remove_table) for y in x]).\
                                        apply(lambda x : [y for y in x if y.isalpha()])
    # Remove stopwords
    if stpw:
        stripped=stripped.apply(lambda x : [y for y in x if not y in stop_words])
    
    pd.set_option('mode.chained_assignment', None)          # Mute assignment warning
    data[txt_key]=stripped.apply(lambda x : ' '.join(x))                                      
    pd.set_option('mode.chained_assignment', 'raise')       # Restore assignment warning
    
    # obtain vocabulary word types 
    types=data[txt_key].str.split(' ', expand=True).stack().unique()   # Vocabulary (word types)
    typesDF=pd.Series(types).to_frame()                                # Data Frame of vocabulary and word embeddings
    typesDF.rename(index=int,columns={0:'Palabra'},inplace=True)
    
    #Add Lemmas to typesDF
    if lematize:
        if lem=='FILE':
            lemas,locs=lematizador(None,'FILE')
            typesDF.drop(labels=locs,axis=0,inplace=True)
            typesDF.loc[:,'Lema']=lemas['Lema']
        else:
            lemas = lematizador(list(types),lem=lem,strt_idx=strt_idx)
            if lemas == None:
                lemas = np.array([None for i in range(len(types))])
            typesDF.loc[:,'Lema']=pd.Series(lemas,index=typesDF.index)
    else:
        lemas = np.array([None for i in range(len(types))])
        typesDF.loc[:,'Lema']=pd.Series(lemas,index=typesDF.index)
    
    #Add Keras-like  embedding placeholders
    krs = np.array([0 for i in range(typesDF.shape[0])])
    typesDF.loc[:,'KRS']=pd.Series(krs,index=typesDF.index)

    #Add W2Vec embedding placeholders
    w2v = np.array([0 for i in range(typesDF.shape[0])])
    typesDF.loc[:,'W2V']=pd.Series(w2v,index=typesDF.index)

    #Add Fast  embedding placeholders
    fst = np.array([0 for i in range(typesDF.shape[0])])
    typesDF.loc[:,'FST']=pd.Series(fst,index=typesDF.index)
   
    return data,typesDF


def view_data(filename=datafile):
    counts=[]
    df = pd.read_csv(filename)
    cols = ['ds'+str(i) for i in range(6)]+['dp0','dp1']
    clases = df.filter(cols)
    categories = list(clases.columns.values)
    for category in categories:
        counts.append((category, clases[category].sum()))
    data_stats = pd.DataFrame(counts, columns=['clase','numero'])
    print(data_stats)   
    data_stats.plot(x='clase', y='numero', kind='bar', legend=False, grid=True, figsize=(10, 8))
    return (df,clases)


def load_data(filename=datafile,lang='spanish',txt_key='Texto',minSize=6,stpw=True,lematize=False,lem='GRAMPAL',stemming=False,prop=0.7,strt_idx=0):
    data = pd.read_csv(filename)
    data,typesDF = preprocess_data(data,lang,txt_key,minSize,stpw,lematize,lem,stemming,strt_idx)
    try:
        if data==None:
            print('data loading error')
            return None,None,None,None,None,None
    except TypeError:
        pass
    data_x = data[txt_key].values
    cols = ['ds'+str(i) for i in range(6)]+['dp0','dp1']
    try:
        data_y = data.filter(cols).values
    except KeyError:
        print(f'No hay clases {cols}')
        print('data loading error')
        return None,None,None,None,None,None
        
    no_paciente=False
    try:
        data_p = data['Paciente'].values
    except KeyError:
        print('Key Error: Paciente')
        no_paciente = True
        print('No habrá datos de Paciente...')

    # split the data, leave 1/3 out for testing
    IDX = np.random.permutation(len(data_x))
    id_split = round(prop * len(IDX))
    x_train, x_test = data_x[IDX[:id_split]], data_x[IDX[id_split:]]
    y_train, y_test = data_y[IDX[:id_split]], data_y[IDX[id_split:]]
    
    if not no_paciente:
        p_train, p_test = data_p[IDX[:id_split]], data_p[IDX[id_split:]]
        savetocsv(p_train, op+"p_train.csv")
        savetocsv(p_test, op+"p_test.csv")
    else:
        p_train=None
        p_test=None
    # Print info
    print(f"Conjunto de entrenamiento con {len(x_train)} instancias.")
    print(f"Conjunto de prueba con {len(x_test)} instancias.")

    # Save to disk
    np.savetxt(op+"y_train.csv", y_train, delimiter=" ")
    np.savetxt(op+"y_test.csv", y_test, delimiter=" ")
    savetocsv(x_train, op+"x_train.csv")
    savetocsv(x_test, op+"x_test.csv")
    return (x_train,y_train),(x_test,y_test),p_train,typesDF

if __name__ == "__main__":
    import sys
    load_data(sys.argv[1])
