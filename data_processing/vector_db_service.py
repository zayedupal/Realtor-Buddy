import json
import os
from pprint import pprint

import chromadb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_processing import POSTGRES_QUERIES

from data_processing.postgres_service import get_connection
from gemini.embedding_gen import embed_text, get_image_embeddings_from_url, \
    get_image_embeddings_from_upload, get_multimodal_embeddings_from_file, \
    get_multimodal_embeddings_from_url, get_image_embeddings_from_file, embed_text_multimodal
from tqdm import tqdm
from dotenv import load_dotenv

from chromadb.utils.data_loaders import ImageLoader
from PIL import Image

load_dotenv()


def get_chroma_persistent_client():
    client = chromadb.PersistentClient(path=os.environ['CHROMA_PERSISTENT_PATH'])
    return client


def select_text_data_from_postgres(connection, limit=None):
    cursor = connection.cursor()
    query = POSTGRES_QUERIES.TEXT_DATA_QUERY if not limit else \
        f"{POSTGRES_QUERIES.TEXT_DATA_QUERY} LIMIT {limit}"
    print(f'query: {query}')
    cursor.execute(query)

    return cursor.fetchall()


def select_precision_data_from_postgres(connection, limit=None):
    cursor = connection.cursor()
    query = POSTGRES_QUERIES.PRECISION_DATA_QUERY if not limit else \
        f"{POSTGRES_QUERIES.PRECISION_DATA_QUERY} LIMIT {limit}"
    print(f'query: {query}')
    cursor.execute(query)

    return cursor.fetchall()


def select_image_data_from_postgres(connection, limit=None):
    cursor = connection.cursor()
    query = POSTGRES_QUERIES.IMG_DATA_QUERY if not limit else \
        f"{POSTGRES_QUERIES.IMG_DATA_QUERY} LIMIT {limit}"
    print(f'query: {query}')
    cursor.execute(query)

    return cursor.fetchall()


def select_multimodal_data_from_postgres(connection, limit=None):
    cursor = connection.cursor()
    query = POSTGRES_QUERIES.IMG_DATA_QUERY if not limit else \
        f"{POSTGRES_QUERIES.IMG_DATA_QUERY} LIMIT {limit}"
    print(f'query: {query}')
    cursor.execute(query)

    return cursor.fetchall()


def create_chroma_collection(name):
    chroma_client = get_chroma_persistent_client()
    collection = chroma_client.get_or_create_collection(name=name)
    return collection


def add_text_in_chroma_collection(collection_name, zpids, documents, meta_data, embeddings=None, batch_size=5000):
    '''
    maximum batch size around 5000
    '''
    chroma_client = get_chroma_persistent_client()
    collection = chroma_client.get_collection(collection_name)
    total = len(zpids)
    total_iterations = total // batch_size
    leftover = total % batch_size

    for i in tqdm(range(total_iterations), desc="upserting text vectors"):
        collection.upsert(
            documents=documents[i * batch_size:(i + 1) * batch_size],
            metadatas=meta_data[i * batch_size:(i + 1) * batch_size],
            ids=[str(zpid) for zpid in zpids[i * batch_size:(i + 1) * batch_size]],
            embeddings=embeddings[i * batch_size:(i + 1) * batch_size] if embeddings else None
        )
    if leftover:
        collection.upsert(
            documents=documents[total_iterations * batch_size:],
            metadatas=meta_data[total_iterations * batch_size:],
            ids=[str(zpid) for zpid in zpids[total_iterations * batch_size:]],
            embeddings=embeddings[total_iterations * batch_size:] if embeddings else None
        )


def add_image_in_chroma_collection(collection_name, zpids, images, meta_data, embeddings=None, batch_size=100):
    chroma_client = get_chroma_persistent_client()
    collection = chroma_client.get_collection(collection_name)
    total = len(zpids)
    total_iterations = total // batch_size
    leftover = total % batch_size

    for i in tqdm(range(total_iterations), desc="upserting image vectors"):
        collection.upsert(
            images=images[i * batch_size:(i + 1) * batch_size],
            metadatas=meta_data[i * batch_size:(i + 1) * batch_size],
            ids=[str(zpid) for zpid in zpids[i * batch_size:(i + 1) * batch_size]],
            embeddings=embeddings[i * batch_size:(i + 1) * batch_size] if embeddings else None
        )
    if leftover:
        collection.upsert(
            images=images[total_iterations * batch_size:],
            metadatas=meta_data[total_iterations * batch_size:],
            ids=[str(zpid) for zpid in zpids[total_iterations * batch_size:]],
            embeddings=embeddings[total_iterations * batch_size:] if embeddings else None
        )

def add_multimodal_in_chroma_collection(collection_name, zpids, images, texts, meta_data, embeddings=None, batch_size=100):
    chroma_client = get_chroma_persistent_client()
    # collection = chroma_client.get_collection(collection_name, embedding_function=gemini_multimodal_embedding_function)
    collection = chroma_client.get_collection(collection_name)
    total = len(zpids)
    total_iterations = total // batch_size
    leftover = total % batch_size

    for i in range(total_iterations):
        collection.add(
            images=images[i * batch_size:(i + 1) * batch_size],
            # documents=texts[i * batch_size:(i + 1) * batch_size],
            metadatas=meta_data[i * batch_size:(i + 1) * batch_size],
            ids=[str(zpid) for zpid in zpids[i * batch_size:(i + 1) * batch_size]],
            embeddings=embeddings[i * batch_size:(i + 1) * batch_size] if embeddings else None
        )
        # collection.add(
        #     # images=images[i * batch_size:(i + 1) * batch_size],
        #     texts=texts[i * batch_size:(i + 1) * batch_size],
        #     metadatas=meta_data[i * batch_size:(i + 1) * batch_size],
        #     ids=[str(zpid) for zpid in zpids[i * batch_size:(i + 1) * batch_size]],
        #     embeddings=embeddings[i * batch_size:(i + 1) * batch_size] if embeddings else None
        # )
    if leftover:
        print(f'images: {images[total_iterations * batch_size:]}')
        collection.add(
            images=images[total_iterations * batch_size:],
            # documents=texts[total_iterations * batch_size:],
            metadatas=meta_data[total_iterations * batch_size:],
            ids=[str(zpid) for zpid in zpids[total_iterations * batch_size:]],
            embeddings=embeddings[total_iterations * batch_size:] if embeddings else None
        )
        # collection.add(
        #     # images=images[total_iterations * batch_size:],
        #     texts=texts[total_iterations * batch_size:],
        #     metadatas=meta_data[total_iterations * batch_size:],
        #     ids=[str(zpid) for zpid in zpids[total_iterations * batch_size:]],
        #     embeddings=embeddings[total_iterations * batch_size:] if embeddings else None
        # )


def search_text_in_chroma_collection(query, collection_name, top_n=10):
    chroma_client = get_chroma_persistent_client()
    query_embeddings = embed_text(texts=[query])
    collection = chroma_client.get_collection(collection_name)
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_n
    )
    return results


def search_image_in_chroma_collection(query_image_path, collection_name, top_n=10, uploaded=False, filter=None):
    chroma_client = get_chroma_persistent_client()
    if uploaded:
        image, query_embeddings = get_image_embeddings_from_upload(query_image_path)
    else:
        image, query_embeddings = get_image_embeddings_from_url(query_image_path)

    collection = chroma_client.get_collection(collection_name)
    if filter:
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=top_n,
            where=filter
        )
    else:
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=top_n
        )
    return results

def search_image_by_text_in_chroma_collection(text, collection_name, top_n=10, filter=None):
    chroma_client = get_chroma_persistent_client()

    query_embeddings = embed_text([text])

    collection = chroma_client.get_collection(collection_name)
    if filter:
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=top_n,
            where=filter
        )
    else:
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=top_n
        )
    return results


def preprocess_detailed_descriptions_ingestion(records):
    zpids = []
    processed_records = []
    meta_data_records = []
    for record in records:
        zpid = record[0]
        combined_text = ""
        meta_data = {}
        for field_inp, value in zip(POSTGRES_QUERIES.DESCRIPTION_FIELDS, record[1:]):
            try:
                field = POSTGRES_QUERIES.DESCRIPTION_DICT[field_inp]
            except:
                field = field_inp
            combined_text += f"{field}:{value}. "
            if field_inp in POSTGRES_QUERIES.METADATA_FIELDS and value:
                meta_data[field_inp] = value
        combined_text += f"link:https://zillow.com/homedetails/{zpid}_zpid. "
        meta_data["link"] = f"https://zillow.com/homedetails/{zpid}_zpid"
        zpids.append(zpid)
        processed_records.append(combined_text)
        meta_data_records.append(meta_data)
    return zpids, processed_records, meta_data_records

def preprocess_precision_data_ingestion(records):
    zpids = []
    processed_records = []
    meta_data_records = []
    for record in records:
        zpid = record[0]
        combined_text = ""
        meta_data = {}
        for field, value in zip(POSTGRES_QUERIES.PRECISION_FIELDS, record[1:]):
            combined_text += f"{field}:{value}. "
            if field in POSTGRES_QUERIES.METADATA_FIELDS and value:
                meta_data[field] = value
        meta_data["link"] = f"https://zillow.com/homedetails/{zpid}_zpid"
        zpids.append(zpid)
        processed_records.append(combined_text)
        meta_data_records.append(meta_data)
    return zpids, processed_records, meta_data_records


def preprocess_image_ingestion(records, img_limit_per_record=None):
    zpids = []
    img_urls = []
    meta_data_records = []
    for record in records:
        zpid = record[0]
        img_urls_json = json.loads(record[1].replace("'", '"'))
        meta_data = {}
        for field, value in zip(POSTGRES_QUERIES.METADATA_FIELDS, record[2:]):
            if value:
                meta_data[field] = value
        meta_data["link"] = f"https://zillow.com/homedetails/{zpid}_zpid"

        img_count = 0

        for i, img_url_json in enumerate(img_urls_json):
            if img_limit_per_record and img_count >= img_limit_per_record:
                break
            # print(f'img_url_json: {img_url_json}')
            meta_data_cur = meta_data.copy()
            img_url = img_url_json['url']
            meta_data_cur['img_url'] = img_url
            # if img_url not in img_urls:
            zpids.append(f'{zpid}_{i}')
            img_urls.append(img_url)
            meta_data_records.append(meta_data_cur)
            img_count += 1
    return zpids, img_urls, meta_data_records


def preprocess_multimodal_ingestion(records, img_limit_per_record=None):
    zpids = []
    descriptions = []
    img_urls = []
    meta_data_records = []
    for record in records:
        # each record contains zpid, image, descriptions
        zpid = record[0]
        img_urls_json = json.loads(record[1].replace("'", '"'))
        meta_data = {}
        combined_text = ""
        for field_inp, value in zip(POSTGRES_QUERIES.DESCRIPTION_FIELDS, record[2:]):
            try:
                field = POSTGRES_QUERIES.DESCRIPTION_DICT[field_inp]
            except:
                field = field_inp
            combined_text += f"{field}:{value}. "
            if field_inp in POSTGRES_QUERIES.METADATA_FIELDS and value:
                meta_data[field_inp] = value

        meta_data["link"] = f"https://zillow.com/homedetails/{zpid}_zpid"

        img_count = 0

        for i, img_url_json in enumerate(img_urls_json):
            if img_limit_per_record and img_count >= img_limit_per_record:
                break
            # print(f'img_url_json: {img_url_json}')
            meta_data_cur = meta_data.copy()
            img_url = img_url_json['url']
            meta_data_cur['img_url'] = img_url
            # if img_url not in img_urls:
            zpids.append(f'{zpid}_{i}')
            img_urls.append(img_url)
            descriptions.append(combined_text)
            meta_data_records.append(meta_data_cur)
            img_count += 1
    return zpids, img_urls, descriptions, meta_data_records


def example_run_text_data():
    # '''
    #     Example Usage
    #     '''
    text_collection_name = os.environ["CHROMA_TEXT_COLLECTION"]
    #
    postgres_connection = get_connection()
    records = select_text_data_from_postgres(postgres_connection, limit=None)
    print(len(records))

    zpids, processed_records, meta_data_records = preprocess_detailed_descriptions_ingestion(records)

    embeddings = embed_text(texts=[desc for desc in processed_records])

    # skip these 2 lines if you already have data in vector storage
    create_chroma_collection(text_collection_name)
    add_text_in_chroma_collection(text_collection_name, zpids, processed_records, meta_data_records, embeddings)
    print("Text Upsert Done..")

    # vector db text query
    # results = search_text_in_chroma_collection("zpid: 14130050", text_collection_name, top_n=1)
    # print(results)


def example_run_precision_data():
    '''
    Example Usage
    '''
    collection_name = os.environ["CHROMA_PRECISION_COLLECTION"]

    postgres_connection = get_connection()
    records = select_precision_data_from_postgres(postgres_connection, limit=None)
    print(len(records))

    zpids, processed_records, meta_data_records = preprocess_precision_data_ingestion(records)

    embeddings = embed_text(texts=[desc for desc in processed_records])

    # skip these 2 lines if you already have data in vector storage
    create_chroma_collection(collection_name)
    add_text_in_chroma_collection(collection_name, zpids, processed_records, meta_data_records, embeddings)

    # vector db text query
    results = search_text_in_chroma_collection("Savannah", collection_name, top_n=10)
    print(results)


def example_run_image_data():
    img_collection_name = os.environ["CHROMA_IMAGE_COLLECTION"]

    postgres_connection = get_connection()
    create_chroma_collection(img_collection_name)

    records = select_image_data_from_postgres(postgres_connection, limit=None)
    print(f'records: {len(records)}')

    zpids, img_urls, meta_data_records = preprocess_image_ingestion(records, img_limit_per_record=3)

    for i, (zpid, img_url, meta_data) in tqdm(enumerate(zip(zpids, img_urls, meta_data_records)),
                                              desc="generating image embeddings", total=len(zpids)):
        try:
            image, embedding = get_image_embeddings_from_url(img_url)

            # skip this line if you already have data in vector storage
            add_image_in_chroma_collection(img_collection_name, [zpid], [image], [meta_data], [embedding])
        except Exception as e:
            print(f'Exception in image embedding generation and storing: {e}')

    print(f'zpids: {len(zpids)}')
    print(f'img_urls: {len(img_urls)}')
    print(f'meta_data_records: {len(meta_data_records)}')

    # vector db text query
    query_image_path = "/Users/zayed/PycharmProjects/Gemini-Real-Estate-App/data_processing/data/images/14131515_1.jpg"
    results = search_image_in_chroma_collection(
        query_image_path,
        img_collection_name,
        top_n=1
    )
    print(results)


def example_run_multimodal_data():
    multimodal_collection_name = os.environ["CHROMA_MULTIMODAL_COLLECTION"]

    postgres_connection = get_connection()
    create_chroma_collection(multimodal_collection_name)

    records = select_multimodal_data_from_postgres(postgres_connection, limit=None)
    print(f'records: {len(records)}')

    zpids, img_urls, descriptions, meta_data_records = preprocess_multimodal_ingestion(records, img_limit_per_record=3)

    for i, (zpid, img_url, description, meta_data) in tqdm(enumerate(zip(zpids, img_urls, descriptions, meta_data_records)),
                                              desc="generating image embeddings", total=len(zpids)):
        # try:
        image, image_embedding, text_embedding = get_multimodal_embeddings_from_url(img_url)
        # embedding = np.concatenate((image_embedding, text_embedding))

        # skip this line if you already have data in vector storage
        add_multimodal_in_chroma_collection(multimodal_collection_name, [zpid], images=[img_url],
                                            texts=[description], meta_data=[meta_data], embeddings=[image_embedding])
        # except Exception as e:
        #     print(f'Exception in image embedding generation and storing: {e}')

    print(f'zpids: {len(zpids)}')
    print(f'img_urls: {len(img_urls)}')
    print(f'meta_data_records: {len(meta_data_records)}')

    # # vector db text query
    # query_image_path = "/Users/zayed/PycharmProjects/Gemini-Real-Estate-App/data_processing/data/images/14131515_1.jpg"
    # results = search_image_in_chroma_collection(
    #     query_image_path,
    #     multimodal_collection_name,
    #     top_n=1,
    #     filter=None
    # )
    # print(results)


def get_all_metadata_df_from_vectordb():
    chroma_client = get_chroma_persistent_client()
    collection = chroma_client.get_collection(os.environ["CHROMA_TEXT_COLLECTION"])
    alldata = collection.get()
    metadatas = alldata['metadatas']
    ids = alldata['ids']
    df = pd.DataFrame(metadatas)
    df['zpid'] = ids
    df['address_city_state'] = df['address_city'] + ", " + df['address_state']

    numeric_columns = ['bathrooms', 'bedrooms', 'price_value']
    # Convert columns to numeric, handling errors with 'errors' argument
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
    print(f'total data: {len(df)}')
    return df


if __name__ == '__main__':
    chroma_client = get_chroma_persistent_client()

    # collection = chroma_client.get_collection(os.environ["CHROMA_MULTIMODAL_COLLECTION"],
    #                                           embedding_function=gemini_multimodal_embedding_function,
    #                                           data_loader=ImageLoader())
    collection = chroma_client.get_collection(os.environ["CHROMA_MULTIMODAL_COLLECTION"])

    ####### Multimodal Tests ###############
    # result = search_image_by_text_in_chroma_collection(text='house with green lawn',
    #                                                    collection_name=os.environ["CHROMA_MULTIMODAL_COLLECTION"],
    #                                                    top_n=3)
    # print(result)

    # result = collection.query(
    #     query_texts=['House with garage'],
    #     include=['documents', 'distances', 'data', 'metadatas', 'uris'],
    #     n_results=3
    # )
    # img = Image.open('/Users/zayed/PycharmProjects/Gemini-Real-Estate-App/data_processing/data/images/70828847_0.jpg')
    # plt.imshow(img)
    # result = collection.query(
    #     query_images=['/Users/zayed/PycharmProjects/Gemini-Real-Estate-App/data_processing/data/images/70828847_0.jpg'],
    #     include=['documents', 'distances', 'data', 'metadatas', 'uris'],
    #     n_results=3
    # )
    # img_response = get_image_embeddings_from_file(image_path='/Users/zayed/PycharmProjects/Gemini-Real-Estate-App/data_processing/data/images/70828847_0.jpg')
    # print(f'img_response: {img_response[1]}')


    # text_response = embed_text_multimodal('House with black roof')
    # print(f'text_response: {text_response}')
    # result = collection.query(
    #     query_embeddings=[text_response],
    #     include=['documents', 'distances', 'data', 'metadatas', 'uris'],
    #     n_results=3
    # )
    # #
    # print(result)
    # # plt.show()
    # #
    # import requests
    # for metadata in result['metadatas'][0]:
    #     cur_img=Image.open(requests.get(metadata['img_url'], stream=True).raw)
    #     link = metadata['link']
    #     print(f'link: {link}')
    #     plt.imshow(cur_img)
    #     plt.show()
    ##################### END OF MULTIMODAL TESTS ####################

    # example_run_text_data()
    # example_run_image_data()

    # chroma_client.delete_collection(os.environ["CHROMA_MULTIMODAL_COLLECTION"])
    example_run_multimodal_data()

    # example_run_precision_data()
    # add_text_in_chroma_collection()
    # get_all_metadata_df_from_vectordb()

    # chroma_client = get_chroma_persistent_client()
    # collection = chroma_client.get_collection(os.environ["CHROMA_TEXT_COLLECTION"])
    # collection.delete('14417385')

