
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
import json
import os
import sys
from datetime import datetime, timedelta
from getopt import getopt


def split_append_chunk(chunk, list):
    chunk_length = len(chunk)
    chunk1 = " ".join(chunk.split()[:chunk_length])
    chunk2 = " ".join(chunk.split()[chunk_length:])
    list.extend([chunk1, chunk2])

def chunk_text(text):
    chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3048,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False
        )
    
    text_chunks = text_splitter.create_documents([text])
    for chunk in text_chunks:
        chunk = chunk.page_content
        count = requests.post(f"{model_service[:-2]}extras/tokenize/count",
                  json={"input":chunk}).content
        count = json.loads(count)["count"]
        if count >= 2048:
            split_append_chunk(chunk, chunks)
        else:
            chunks.append(chunk)
    
    return chunks

def read_file(file):
    file = open(file, "r")
    content = file.read()
    file.close()
    return content

model_service = "http://localhost:8001/v1"
file = "fake_meeting.txt"

opts, args = getopt(sys.argv[1:],'p:f:',['port=','file='])
for option, argument in opts:
    if option == '-p':
        model_service = f"http://localhost:{argument}/v1"
    if option == '-f':
        file = argument

minDuration = timedelta.max
maxDuration = timedelta.min
totalDuration = timedelta()


llm = ChatOpenAI(base_url=model_service,
             api_key="not required",
             streaming=False,
             temperature=0.0,
             max_tokens=400,
             )

### prompt example is from  https://python.langchain.com/docs/use_cases/summarization
refine_template = PromptTemplate.from_template(
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary"
    "If the context isn't useful, return the original summary."
    "Only use bullet points."
    "Dont ever go beyond 10 bullet points."
)


if file != None:

    text = read_file(file)
    chunks = chunk_text(text)
    num_chunks = len(chunks)
    print(f"Processing data in {num_chunks} chunks...")
    existing_answer = ""
    
    for i, chunk in enumerate(chunks):
        print(f"Sending request {i}")
        start = datetime.now()
        response = llm.invoke(refine_template.format(text=chunk,existing_answer=existing_answer))
        duration = datetime.now() - start
        if duration < minDuration:
            minDuration = duration
        if duration > maxDuration:
            maxDuration = duration
        totalDuration += duration
        print(f"Duration {duration}")
        existing_answer = response.content

    print(f"Final response: {response}")
    print(f"min {minDuration} max {maxDuration} total {totalDuration}")

