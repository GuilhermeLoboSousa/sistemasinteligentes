import sys
sys.path.append("C:\\Users\\guilh\\OneDrive\\Documentos\\GitHub\\sistemasinteligentes")
import numpy as np

class OneHotEncoder:
    '''
    One-hot encoder for sequences.
    '''

    def __init__(self, padder:str,max_length:int=None):
        """
        Parameters
        ----------
        padder:
            character to perform padding with
        max_length:
            maximum length of sequences
        
        Attributes
        -----------
        alphabet:
            the unique characters in the sequences
        char_to_index:dict
            dictionary mapping characters in the alphabet to unique integers
        index_to_char:dict
            reverse of char_to_index (dictionary mapping integers to characters)
        """
        self.padder=padder
        self.max_lenght=max_length

        self.alphabet=set()
        self.char_to_index={}
        self.index_to_char={}
    
    def fit (self,data:list[str]):
        """
        Parameters
        ---------
        data:list[str]
            list of sequences to learn from
        Returns
        -------
        OneHotEncoder
            The fitted encoder.
        """
        #max_lenght não é fornecido
        if self.max_lenght is None:
            tamanhos=[]
            for sequence in data:
                tamanho=len(sequence)
                tamanhos.append(tamanho)
            self.max_lenght=np.max(tamanhos) #ficamos a saber o tamanaho da maior sequencia nos dados

        #saber o alfabeto que estou a trabalhar
        all_seq="".join(data)#junto todas para depois conseguir ir buscar apenas caracter que sao diferenciados
        self.alphabet=np.unique(list(all_seq))#vou ter apenas os diferenciados
        #alfabeto neste momento sera uma lista com os caracteres que lhe pertenem , eg, ["a","b","c"]

        #char_to index e index_to char
        indices = np.arange(1,len(self.alphabet)+1) #quero que tenha associado um numero a uma letra que comece em 1 e va até ao tamanho do alfabeto
        #se tiver 5 letras vai de 1 a 5(o +1 tem haver com python)
        #para poder fazer associado a:1 e 1:a
        self.char_to_index = dict(zip(self.alphabet, indices))
        self.index_to_char = dict(zip(indices,self.alphabet))#como quero o inverso fica facil

        #vi me obrigado a fazer isto porque cause do padling porque caso contrario nao teria esse caracter de pad no dicionario e dava erro
        if self.padder not in self.alphabet:
            self.alphabet = np.append(self.alphabet, self.padder)
            max_index = max(self.char_to_index.values())
            new_index = max_index + 1
            self.char_to_index[self.padder] = new_index
            self.index_to_char[new_index] = self.padder
        #ou seja estou a acrecentar o meu "?" ao dicionario
                
        return self
    
    def transform(self, data:list[str]) ->np.ndarray:
        """
        Parameter
        ---------
        data:list[str]
            data to encode
        
        Returns
        --------
        np.ndarray:
            One-hot encoded matrices
        """
        #trim a sequencia até a o maxio de comprimento
        sequence_trim_pad=[]
        for sequence in data:
            trim_pad=sequence[:self.max_lenght].ljust(self.max_lenght, self.padder)#tive ajuda do chat aqui pq nao conhecia o ljust que permite faer exatamente o que queria; coloquei ? para ser mais visual em vez de " "
            sequence_trim_pad.append(trim_pad) 
        #fico com as minhas seq com o mesmo tamanho, as que forem pequenas serao acrecentados "?"
        
        #agora criar a encode onde no fim terei uma matriz para cada sequencia de max_length * alfabeto-o que faz sentido
        one_hot_encode=[]
        matriz_identidade=np.eye(len(self.alphabet))
        print(matriz_identidade)
        #criar uma matriz identidade com o tamamho do meu alfabeto , ou seja [1,0,0][0,1,0],[0,0,1] isto para um exemlo de alfabeto com 3 letras/caracteres
        #ou seja fico com matriz quadrada onde as colunas sao o alfabeto (que é importante) e cada letra vai ter preenchido um 1
        for sequence_ajustada in sequence_trim_pad:#vai a cada sequencia que ja colcoada todas com o mesmo tamanho
            for letra in sequence_ajustada:#vai letra a letra
                value_no_dicionario=self.char_to_index.get(letra)
                one_hot_sequence=matriz_identidade[value_no_dicionario-1]
                #-1 por causa dos index em python caso contrario da erro
                #para ficar algo por exemplo letra A: [1,0,0]; letra B ja vou buscar a 2 linha da matriz identidade [0,1,0] etc
                #isto pq antes ja sei o indice de cada letra logo consigo ir buscar a linha da matriz identidade que vai ter o 1 apenas na letra do alfabeto que eu quero
                #tive de fazer np.array para depois ter uma matriz do genero da one hot encoding
                one_hot_encode.append(one_hot_sequence)
        return one_hot_encode#problema dos array e merdas

    def fit_transform(self, data: list[str]) -> np.ndarray:
        """
        Parameters
        ----------
        data: list[str]
            list of sequences to learn from
        Returns
        -------
        np.ndarray:
            One-hot encoded matrices
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> list[str]:
        """
        Parameters
        ----------
        data: np.ndarray-vem de cima
            one-hot encoded matrices to decode
        Returns
        -------
        list[str]:
            Decoded sequences
        """
        #vamos ficar com nos indices que têm valor 1 na matriz hot_encoding
        index=[]
        for one_hot_matrix in data:
            indices = np.argmax(one_hot_matrix)
            index.append(indices)
        total_sequences=[]
        for each_index in index:
            letras=self.index_to_char.get(each_index+1)#mais um por ser python
            total_sequences.append(letras)
            texto="".join(total_sequences)#vamos ficar com uma string com todos os caracteres de todas as sequencias
        final=[]
        for i in range (0,len(texto),self.max_lenght):#vai agora analisar cada trecho de sequencia de cada vez
            string=texto[i:i+self.max_lenght]
            corte_extra=string.rstrip(self.padder)#cortar aquilo que foi acrescentado para todas as seq ficarem com memso tamanho
            final.append(corte_extra)
        return final #decoded sequences
#testar
encoder = OneHotEncoder(padder="?", max_length=9)
data = ["abc", "aabd"]
encoded_data = encoder.fit_transform(data)
print("One-Hot Encoding:")
print(encoded_data)
decoded_data = encoder.inverse_transform(encoded_data)

print("\nDecoded Data:")
print(decoded_data)
