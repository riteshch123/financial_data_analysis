# Description: This script fetches the content from the PDFs and stores it in Pinecone
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
from connections import pinecone_connection, openai_connection
from openai import OpenAI
import os



def fetch():
    # load PDFs
    loaders = [
        PyPDFLoader("NVDA_2023Q1.pdf"),
        PyPDFLoader("NVDA_2023Q2.pdf"),
        PyPDFLoader("NVDA_2023Q3.pdf")
    ]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())

    # Construct the DataFrame
    data = {
        "id": list(range(len(pages))),
        "text": [doc.page_content for doc in pages],
        "source": [doc.metadata['source'] for doc in pages],
        "page_no": [doc.metadata['page'] for doc in pages]
    }

    df = pd.DataFrame(data)

    return df


def storing_pinecone():
    '''
    function to store set a in pinecone
    '''
    try:
        df = fetch()

        # openai
        api_key = os.environ.get('OPENAI_API_KEY')
        openai_client = OpenAI(api_key=api_key)

        # Pinecone
        pinecone_api_key, index_name = pinecone_connection()
        pinecone = Pinecone(api_key=pinecone_api_key)
        index = pinecone.Index(name=index_name)

        all_embeddings = []

        # iterating over pandas dataframe
        print("Generating embeddings..")
        for _, row in df.iterrows():
            _id = str(row['id'])
            text = row['text']
            source = row['source']
            page_no = row['page_no']

            # embedding data
            embedding = openai_client.embeddings.create(
                input=text,
                model='text-embedding-ada-002',
            ).data[0].embedding

            print(f"Embedded data for {_id}")

            embedding_data = {
                "id": _id,
                "values": embedding,
                "metadata": {
                    "id": _id,
                    "file_name": source,
                    "page_no": page_no,
                    "text": text
                }
            }

            # embedding question and answer separately
            all_embeddings.append(embedding_data)

        print("Embeddings generated")

        # upserting the embeddings to pinecone namespace
        index.upsert(all_embeddings)

        return "successful"

    except Exception as e:
        print("Exception in storing_pinecone() function: ", e)
        return "failed"
