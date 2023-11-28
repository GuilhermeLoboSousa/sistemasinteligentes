import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np
# teste com o verdadeiro modelo do sckit learn

import numpy as np
from collections import Counter
import random
v=["abc","aabd"]
alfa=set()
tamanhos=[]
for x in v:
    tamanho=len(x)
    tamanhos.append(tamanho)
a=np.max(tamanhos)
#print(a)
all="".join(v)
alfa=np.unique(list(all))
indices = np.arange(1,len(alfa)+1)
char_to_i=dict(zip(alfa,indices))
i_to_char=dict(zip(indices,alfa))
if "?" not in alfa:
    alfa = np.append(alfa, "?")
    max_index = max(char_to_i.values())
    new_index = max_index + 1
    char_to_i["?"] = new_index
    i_to_char[new_index] = "?" 
sequence_trim_pad=[]
for sequence in v:
    trim_pad=sequence[:4].ljust(4, "?")#tive ajuda do chat aqui pq nao conhecia o ljust que permite faer exatamente o que queria; coloquei ? para ser mais visual em vez de " "
    sequence_trim_pad.append(trim_pad)
one_hot_encode=[]
matriz_identidade=np.eye(len(alfa))
for seq_ajust in sequence_trim_pad:
    for letra in seq_ajust:
        value_nodic=char_to_i.get(letra)
        vetor=matriz_identidade[value_nodic-1]
        one_hot_encode.append(vetor)
print(one_hot_encode)
index=[]
for f in one_hot_encode:
    indices=np.argmax(f)
    index.append(indices)
sequences=[]

for a in range(0,len(index),4):
    grupo=index[a:a+4]
    for e in grupo:
        final=i_to_char.get(e+1)
        teste="".join(final)
print(teste)
