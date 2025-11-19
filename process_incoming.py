import json
import pandas as pd
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib



def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    embedding = r.json()["embeddings"] 
    return embedding

def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        #"model": "deepseek-r1",
        "model": "llama3.2:1b",
        "prompt": prompt,
        "stream": False
    })
    response= r.json()
    #print(response)
    return response


df= joblib.load('embeddings.joblib')

incoming_query= input("Ask your question:")
question_embedding= create_embedding([incoming_query])[0]

similarities= cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
#print(similarities)
top_result= 5
max_indx= similarities.argsort()[::-1][0:top_result]
new_df= df.iloc[max_indx]
#print(new_df[['text', 'index']])

prompt= f'''These are video playlist of YC Combinator on How to start a Startup. Here are video chunks containing lecture number, index and text of things said in lecture:
{new_df[['text', 'index']].to_json(orient="records")}
-----------------------------------------------------------------------------------------------------------------
"{incoming_query}"
User asked this question related to the video chunks, you have to answer in a human way (don't mention the above format, it's just for you) where and how much content is taught in which lecture and guide the user to go to that particular video. If user asks unrelated questions, tell him/her that you can answer only questions related to the course.
'''
with open('prompt.txt', 'w') as f:
    f.write(prompt)
    
response= inference(prompt)
print(response["response"])

with open('response.txt', 'w') as f:
    f.write(response["response"])    
# for index, item in new_df.iterrows():
#     print(index, item['text'], item['source'].split('/')[-1].split('.')[0])