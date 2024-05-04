import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI
from streamlit_extras.stylable_container import stylable_container

from Realtor_Buddy import authentication
from css_codes import file_uploader_css
from data_processing.vector_db_service import get_chroma_persistent_client, get_all_metadata_df_from_vectordb
from gemini.embedding_gen import embed_text_multimodal, \
    get_image_embeddings_from_upload

load_dotenv()


def real_estate_multimodal_llm_agent_with_manual_input(llm):
    template = """
    Format the house information in a concise human readable format with bullte points if possible.
    Output format should be: 
    address
    features
    'link'
    An example 'link' could be: 'https://zillow.com/homedetails/307932169_zpid'  from the given context ONLY.

        context: {context}

    """

    promptllm = PromptTemplate(
        template=template,
        input_variables=["context"]
    )

    llm_chain = promptllm | llm

    return llm_chain


def get_mandatory_information():
    mandatory_info = """Location, Price_range, House_type, Number_of_beds, Number_of_baths"""
    return mandatory_info


def convert_chat_history_to_prompt_string(history) -> str:
    print(f'history: {history}')
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])


def summarize_chat_history_llm_agent(llm):
    summary_prompt_tmplt = """Summarize the chat history within 30 words and keep the important information. 
    Show the summary only, nothing else
    chat_history: {chat_history}"""

    promptllm = PromptTemplate(
        template=summary_prompt_tmplt,
        input_variables=["chat_history"]
    )

    llm_chain = promptllm | llm

    return llm_chain

def format_vector_db_mm_result(query_result):
    ids = query_result['ids'][0]
    distances = query_result['distances'][0]
    metadatas = query_result['metadatas'][0]

    return ids, distances, metadatas


def get_llm_response(user_input, chat_history_details):
    with st.spinner('Searching for relevant properties...'):
        # LLM model
        llm = VertexAI(model_name=os.environ["LLM_MODEL"])

        print(f'uploaded image: {st.session_state["uploaded_image"]!=None}')
        if st.session_state["uploaded_image"]:
            # # Get chat history summary
            # chat_summary_chain = summarize_chat_history_llm_agent(llm)
            # chat_history = chat_summary_chain.invoke({'chat_history': chat_history_details})
            # st.session_state['chat_history'] = chat_history

            _, image_embedding = get_image_embeddings_from_upload(st.session_state["uploaded_image"].read())

            chroma_client = get_chroma_persistent_client()
            collection = chroma_client.get_collection(os.environ["CHROMA_MULTIMODAL_COLLECTION"])
            query_result = collection.query(
                query_embeddings=[image_embedding],
                include=['documents', 'distances', 'data', 'metadatas', 'uris'],
                n_results=3,
                where={"link": {"$in": [f"https://zillow.com/homedetails/{zpid}_zpid" for zpid in
                                        st.session_state['zpids']]}}
            )
            st.session_state["uploaded_image"] = None
        else:
            print("I am herreeeeee")
            # Get chat history summary
            chat_summary_chain = summarize_chat_history_llm_agent(llm)
            chat_history = chat_summary_chain.invoke({'chat_history': user_input})
            st.session_state['chat_history'] = chat_history


            text_response = embed_text_multimodal(user_input)


            chroma_client = get_chroma_persistent_client()
            collection = chroma_client.get_collection(os.environ["CHROMA_MULTIMODAL_COLLECTION"])
            query_result = collection.query(
                query_embeddings=[text_response],
                include=['documents', 'distances', 'data', 'metadatas', 'uris'],
                n_results=3,
                where={"link": {"$in": [f"https://zillow.com/homedetails/{zpid}_zpid" for zpid in
                                         st.session_state['zpids']]}}
            )

        print(f"query_result: {query_result}")

        chat_chain = real_estate_multimodal_llm_agent_with_manual_input(llm)

        ids, distances, metadatas = format_vector_db_mm_result(query_result)

        responses = []
        similar_images = []
        for id, distance, metadata in zip(ids, distances, metadatas):
            similar_img = metadata['img_url']
            response = chat_chain.invoke({'context': metadata})
            responses.append(response)
            similar_images.append(similar_img)
        print(f'responses: {responses}')
        print(f'similar_images: {similar_images}')

    return responses, similar_images


def st_mandatory_info_collection():
    all_metadata = get_all_metadata_df_from_vectordb() if "all_metadata" not in st.session_state else st.session_state.all_metadata
    st.session_state["all_metadata"] = all_metadata
    loc = st.selectbox(label="Select Location", options=all_metadata['address_city_state'].unique())
    propertytype = st.multiselect(label="Select Property Type",
                                  options=all_metadata[all_metadata['address_city_state'] == loc][
                                      'propertytype'].unique())
    price = st.slider(label="Select Price Range",
                      value=(300000.00, 500000.0), step=10000.0,
                      min_value=all_metadata[all_metadata['address_city_state'] == loc]['price_value'].min(),
                      max_value=all_metadata[all_metadata['address_city_state'] == loc]['price_value'].max())
    bedrooms = st.slider(label="Select Number of Bedrooms",
                         value=(2.0, 4.0), step=1.0,
                         min_value=all_metadata[all_metadata['address_city_state'] == loc]['bedrooms'].min(),
                         max_value=all_metadata[all_metadata['address_city_state'] == loc]['bedrooms'].max())
    bathrooms = st.slider(label="Select Number of Bathrooms",
                          value=(2.0, 4.0), step=1.0,
                          min_value=all_metadata[all_metadata['address_city_state'] == loc]['bathrooms'].min(),
                          max_value=all_metadata[all_metadata['address_city_state'] == loc]['bathrooms'].max())

    if st.button("Submit"):
        cur_filter = ((all_metadata['address_city_state'] == loc) & (all_metadata['propertytype'].isin(propertytype)) & \
                     (all_metadata['price_value'] >= price[0]) & (all_metadata['price_value'] <= price[1]) & \
                     (all_metadata['bathrooms'] >= bathrooms[0]) & (all_metadata['bathrooms'] <= bathrooms[1]) & \
                     (all_metadata['bedrooms'] >= bedrooms[0]) & (all_metadata['bedrooms'] <= bedrooms[1]))

        st.session_state['zpids'] = all_metadata[cur_filter]['zpid'].tolist()
        st.session_state['location'] = loc
        st.session_state['propertytype'] = propertytype
        st.session_state['price'] = price
        st.session_state['bedrooms'] = bedrooms
        st.session_state['bathrooms'] = bathrooms


def get_user_details_and_show_specific_data(user_email):
    print(f"Logged in User: {user_email}")
    pass

def is_http_url(url_string):
  """Checks if a string starts with 'http://' or 'https://'."""
  valid_prefixes = ("http://", "https://")
  return url_string.startswith(valid_prefixes)

def main():
    st.set_page_config("Realtor Buddy Multimodal Chat", layout="wide")
    st.header("Realtor Buddy Multimodal Chat")

    if "auth" not in st.session_state:
        authentication()
    else:
        uploaded_image = None
        # The main UI functionality starts from here
        with st.sidebar:
            st_mandatory_info_collection()

        # prompt = st.chat_input(disabled=("location" not in st.session_state))
        prompt = st.chat_input()

        st.session_state["uploaded_image"] = None
        chat_col, summary_col = st.columns([5, 1])

        with summary_col:
            with stylable_container(
                    key="bottom_content_2",
                    css_styles="""
                        {
                            position: fixed;
                            bottom: 120px;
                        }
                        """,
            ):
                st.markdown(file_uploader_css, unsafe_allow_html=True)
                st.session_state["uploaded_image"] = st.file_uploader("Upload Image")
        with chat_col:
            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                    {
                        "role": "assistant",
                        "content": "Hey there! I'm your real estate sidekick. Let's find your dream house today! "
                                   "Please select the options on the left to start your search. "
                                   "Tell us more about the look of the houses or upload in image. "
                                   "I'll try my best to pull up relevant listings for you!"
                    }
                ]
            for msg in st.session_state.messages:
                if is_http_url(msg["content"]):
                    with st.chat_message(msg["role"]):
                        st.image(msg["content"])
                else:
                    st.chat_message(msg["role"]).write(msg["content"])

            if ("location" in st.session_state):
                if prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.chat_message("user").write(prompt)

                    ai_responses, img_responses = get_llm_response(prompt, st.session_state.messages)

                    if not ai_responses and not img_responses:
                        st.chat_message("assistant").write("Sorry, we don't have any relevant listings to recommend. Can you please try other filters on the left panel?")
                    for ai_response, img_response in zip(ai_responses, img_responses):
                        with st.chat_message("assistant"):
                            st.image(img_response)
                            st.write(ai_response)

                            st.session_state.messages.append({"role": "assistant", "content": img_response})
                            st.session_state.messages.append({"role": "assistant", "content": ai_response})

        # with summary_col:
        #     with stylable_container(
        #             key="bottom_content",
        #             css_styles="""
        #                 {
        #                     position: fixed;
        #                     bottom: 240px;
        #                 }
        #                 """,
        #     ):
        #         st.markdown('**Chat summary so far**')
        #         if 'chat_history' in st.session_state:
        #             st.write(st.session_state['chat_history'])


if __name__ == "__main__":
    main()
