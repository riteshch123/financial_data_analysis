from openai import OpenAI
from pinecone import Pinecone
from connections import openai_connection, pinecone_connection


def fetch_from_pinecone(text):
    '''
    function to fetch data from pinecone
    '''
    try:
        # Pinecone
        pinecone_api_key, index_name = pinecone_connection()
        pinecone = Pinecone(api_key=pinecone_api_key)

        # openai
        key = openai_connection()
        model = OpenAI(api_key=key)

        # fetching data from pinecone
        index = pinecone.Index(name=index_name)

        xq = model.embeddings.create(
            input=text,
            model='text-embedding-ada-002',
        ).data[0].embedding

        xc = index.query(vector=xq, top_k=3, include_metadata=True)

        for match in xc['matches']:
            score = match['score']
            text = match['metadata']['text']
            match = f"{round(score, 2)}: {text}"

        return match

        # fetching data from pinecone namespace
    except Exception as e:
        print("Exception in fetch_from_pinecone() function: ", e)
        return "failed"


def generate_response(question: str):

    # openai
    key = openai_connection()
    openai_client = OpenAI(api_key=key)
    context = fetch_from_pinecone(question)

    # prompt to openai for generating analysis of sample QA
    prompt = f"Answer the question based only based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\n"

    print("generating response...")
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.5,
        # max_tokens=500,
    )

    print("response generated")

    # Extract and return the generated analysis from the response
    analysis = response.choices[0].message.content
    print(analysis)

    return analysis
