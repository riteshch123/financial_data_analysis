import configparser
config = configparser.ConfigParser()
config.read('configuration.properties')

def pinecone_connection():
    try:
        pinecone_api_key = config['PINECONE']['PINECONE_API_KEY']
        index_name = config['PINECONE']['INDEX']
        return pinecone_api_key, index_name
    except Exception as e:
        print("Exception in pinecone_connection function: ", e)
        return


def openai_connection():
    try:
        openai_api_key = config['OPENAI']['OPENAI_API_KEY']
        return openai_api_key
    except Exception as e:
        print("Exception in openai_connection function: ", e)
        return
