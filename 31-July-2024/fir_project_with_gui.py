import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import pandas as pd

def preprocess_text(text):
    text=text.lower()
    words=word_tokenize(text)
    stop_word=set(stopwords.words('english'))
    words=[word for word in words if word not in stop_word]
    stemmer=PorterStemmer()
    words=[stemmer.stem(word) for word in words]

    proprocessed_text=' '.join(words)
    return proprocessed_text


import pickle
new_ds = pickle.load(open('D:/My WorkSpace/Summer-Internship/29-July-2024/preprocess_data.pkl'))

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer,util
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


def suggest_sections(complaint,dataset,min_suggestions=5):
    preprocessed_complaint=preprocess_text(complaint)
    complaint_embedding=model.encode(preprocessed_complaint)
    section_embedding=model.encode(dataset['Combo'].tolist())
    similarities=util.pytorch_cos_sim(complaint_embedding,section_embedding)[0]
    similarity_threhold=0.2
    relevant_indices=[]
    while len(relevant_indices)<min_suggestions and similarity_threhold>0:
        relevant_indices=[i for i, sim in enumerate(similarities)if sim>similarity_threhold]
        similarity_threhold-=0.5 #st=st-0.5
        sorted_indices=sorted(relevant_indices,key=lambda i: similarities[i],reverse=True)
        suggestions=dataset.iloc[sorted_indices][['Description','Offense','Punishment','Cognizable','Bailable','Court','Combo']].to_dict(orient='records')
        return suggestions
    
    
complaint=input("Enter crime description")
suggest_sections=suggest_sections(complaint,new_ds)
if suggest_sections:
    print("Suggested Section are:")
    for suggestion in suggest_sections:
        print(f"Description: {suggestion['Description']}")
        print(f"Offense: {suggestion['Offense']}")
        print(f"Punishment: {suggestion['Punishment']}")
        print("_________________________________________________________________________________________\n")
else:
    print("No record is found")
