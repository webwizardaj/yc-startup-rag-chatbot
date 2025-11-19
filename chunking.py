import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

for file in os.listdir("transcript"):
    if file.endswith(".txt"):
        transcript_file = f"transcript/{file}"
        json_file = f"transcript/{file.replace('.txt', '.json')}"
        try: 
            with open(transcript_file, "r", encoding="utf-8") as f:
                transcript_text = f.read()
        except FileNotFoundError:
            print(f"File {transcript_file} not found")
            exit(1)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = splitter.split_text(transcript_text)

        json_data = []
        for i, chunk_text in enumerate(chunks):
            json_data.append({"index": i, "text": chunk_text, "source": transcript_file})
            
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
            
        print(f"Successfully created {len(json_data)} chunks and saved to {json_file}")

transcript_file = "transcript/lecture1.txt"
json_file = "transcript/lecture1.json"

try: 
    with open(transcript_file, "r", encoding="utf-8") as f:
        transcript_text = f.read()
except FileNotFoundError:
    print(f"File {transcript_file} not found")
    exit(1)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
chunks = splitter.split_text(transcript_text)

json_data = []
for i, chunk_text in enumerate(chunks):
    json_data.append({"index": i, "text": chunk_text, "source": transcript_file})
    
with open(json_file, "w", encoding="utf-8") as f:
    json.dump(json_data, f, ensure_ascii=False, indent=4)
    
print(f"Successfully created {len(json_data)} chunks and saved to {json_file}")