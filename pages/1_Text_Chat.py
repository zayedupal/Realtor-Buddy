import os
from pprint import pprint

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI
from streamlit_extras.stylable_container import stylable_container

from data_processing.vector_db_service import get_chroma_persistent_client, get_all_metadata_df_from_vectordb
from gemini.embedding_gen import gemini_text_embedding_function

load_dotenv()


def real_estate_llm_agent_with_manual_input(llm):
    template = """
    You are a real-estate property assistant.
    Your goal is to find out properties or listings that match best with the chat_history and provided context.
    Show at most 3 most relevant properties.

    Based on the chat_history do the following in order:
     - You MUST show summary of the relevant properties in bullet points
     - You MUST show the informations mentioned in chat_history for each of the properties
     - You MUST show the ZPID for each house from the given context ONLY
     - You MUST show the web link for each house from the given context ONLY 
     - Generate the link from the 'link' property of the context. An example could be 'link': 'https://zillow.com/homedetails/307932169_zpid'  from the given context ONLY
     - Please ONLY generate the link from the 'link' property of the context. An example could be 'link': 'https://zillow.com/homedetails/307932169_zpid'
     - If you don't have result in the context, then don't show result and apologise. Recommend user to update the filters
     - Ask for some more questions related to property buying from the user. You MUST not ask for information that you already have
    Never recommend houses that you don't have in the context. If you don't have it politely say that you don't have any relevant listings
    You MUST show the response in a good human readable format.
    
        chat_history:{chat_history}

        context: {context}

    """

    promptllm = PromptTemplate(
        template=template,
        input_variables=["chat_history", "context"]
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
    summary_prompt_tmplt = """Summarize the chat history and make sure you keep the most important information
    provided by the user. DON'T include assistant's chat.
    Keep the summary within 50 words. Show the summary only, without bolding out any text.
    chat_history: {chat_history}"""

    promptllm = PromptTemplate(
        template=summary_prompt_tmplt,
        input_variables=["chat_history"]
    )

    llm_chain = promptllm | llm

    return llm_chain


def get_llm_response(user_question, chat_history_details, stream=True):
    with st.spinner('Searching for relevant properties...'):
        # LLM model
        llm = VertexAI(model_name=os.environ["LLM_MODEL"])

        # Get chat history summary
        chat_summary_chain = summarize_chat_history_llm_agent(llm)
        chat_history = chat_summary_chain.invoke({'chat_history': chat_history_details})
        st.session_state['chat_history'] = chat_history

        # Context Building: Perform similarity search in the vector database based on the user question
        # docs = search_text_in_chroma_collection(user_question, os.environ["CHROMA_TEXT_COLLECTION"], top_n=10)
        langchain_chroma = Chroma(
            client=get_chroma_persistent_client(),
            collection_name=os.environ["CHROMA_TEXT_COLLECTION"],
            embedding_function=gemini_text_embedding_function,
        )

        context = langchain_chroma.search(
            query=chat_history, search_type="similarity", k=5,
            filter={"link": {"$in": [f"https://zillow.com/homedetails/{zpid}_zpid" for zpid in
                                     st.session_state['zpids']]}}
        )

        chat_chain = real_estate_llm_agent_with_manual_input(llm)
        if stream:
            response = chat_chain.stream({'chat_history': chat_history, 'context': context})
        else:
            response = chat_chain.invoke(
                {'human_input': user_question, 'mandatory_information': get_mandatory_information(),
                 'chat_history': chat_history, 'context': context})

        print(
            f"session vals: {st.session_state['location'], st.session_state['propertytype'], st.session_state['price'], st.session_state['bedrooms'], st.session_state['bathrooms']}")
        print(f"context metadata:")
        pprint([c.metadata for c in context])
        print("========================================================")
        print(f"chat_history: {chat_history}")
    return response


def st_mandatory_info_collection():
    all_metadata = get_all_metadata_df_from_vectordb() if "all_metadata" not in st.session_state else st.session_state.all_metadata
    st.session_state["all_metadata"] = all_metadata
    loc = st.selectbox(label="Select Location", options=all_metadata['address_city_state'].unique(), index=4)
    propertytype = st.multiselect(label="Select Property Type",
                                options=all_metadata[all_metadata['address_city_state'] == loc][
                                    'propertytype'].unique(), default=['singleFamily', 'condo'])
    price = st.slider(label="Select Price Range",
                      value=(300000.00, 520000.0), step=10000.0,
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

        prompt = (f"I am looking for {propertytype} house in {loc} "
                  f"with {bedrooms[0]} to {bedrooms[1]} bedrooms and "
                  f"{bathrooms[0]} to {bathrooms[1]} bathrooms. "
                  f"My budget is within {price[0]} to {price[1]}.")

        mandatory_info_message = {
                "role": "user",
                "content": prompt
            }
        if "messages" not in st.session_state:
            st.session_state.messages = [mandatory_info_message]
        else:
            st.session_state.messages.append(mandatory_info_message)
        st.session_state.messages.append({"role": "assistant", "content": get_llm_response(prompt, st.session_state.messages, stream=False)})

def get_user_details_and_show_specific_data(user_email):
    print(f"Logged in User: {user_email}")
    pass

def main():
    st.set_page_config("Realtor Buddy Text Chat", layout="wide")
    st.header("Realtor Buddy Text Chat")

    # TODO: Uncomment once the SSO issue is fixed
    CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
    CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
    AUTHORIZE_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
    REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"

    # The main UI functionality starts from here
    with st.sidebar:
        st_mandatory_info_collection()

    prompt = st.chat_input(disabled=("location" not in st.session_state))

    chat_col, summary_col = st.columns([5, 1])
    with chat_col:
        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": "Hey there! I'm your real estate sidekick. Let's find your dream house today! "
                               "Please select the options on the left to start your search."
                }
            ]
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])
        if ("location" in st.session_state):
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.chat_message("user").write(prompt)

                response = st.chat_message("assistant").write_stream(
                    get_llm_response(prompt, st.session_state.messages))
                st.session_state.messages.append({"role": "assistant", "content": response})

    with summary_col:
        with stylable_container(
                key="bottom_content",
                css_styles="""
                    {
                        position: fixed;
                        bottom: 240px;
                    }
                    """,
        ):
            st.markdown('**Chat summary so far**')
            if 'chat_history' in st.session_state:
                st.write(st.session_state['chat_history'])

    # if "auth" not in st.session_state:
    #     # create a button to start the OAuth2 flow
    #     oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_ENDPOINT, TOKEN_ENDPOINT, TOKEN_ENDPOINT,
    #                              REVOKE_ENDPOINT)
    #     result = oauth2.authorize_button(
    #         name="Continue with Google",
    #         icon="https://www.google.com.tw/favicon.ico",
    #         # TODO: Read from env file the redirect uri
    #         redirect_uri="http://localhost:8501",
    #         scope="openid email profile",
    #         key="google",
    #         extras_params={"prompt": "consent", "access_type": "offline"},
    #         use_container_width=True,
    #         pkce='S256',
    #     )
    #
    #     if result:
    #         print(result)
    #         st.write(result)
    #         # decode the id_token jwt and get the user's email address
    #         id_token = result["token"]["id_token"]
    #         # verify the signature is an optional step for security
    #         payload = id_token.split(".")[1]
    #         # add padding to the payload if needed
    #         payload += "=" * (-len(payload) % 4)
    #         payload = json.loads(base64.b64decode(payload))
    #         email = payload["email"]
    #         print(email)
    #
    #         st.session_state["auth"] = email
    #         st.session_state["token"] = result["token"]
    #         st.rerun()
    # else:
    #     st.write("You are logged in!")
    #     st.write(st.session_state["auth"])
    #
    #     email = st.session_state["auth"]
    #     get_user_details_and_show_specific_data(email)
    #
    #     # st.write(st.session_state["token"])
    #     st.button("Logout")
    #     del st.session_state["auth"]
    #     del st.session_state["token"]
    #
    #     # The main UI functionality starts from here
    #     with st.sidebar:
    #         st_mandatory_info_collection()
    #
    #     prompt = st.chat_input(disabled=("location" not in st.session_state))
    #
    #     if "messages" not in st.session_state:
    #         st.session_state["messages"] = [
    #             {
    #                 "role": "assistant",
    #                 "content": "Hey there! I'm your real estate sidekick. Let's find your dream house today! "
    #                            "Please select the options on the left to start your search."
    #             }
    #         ]
    #     for msg in st.session_state.messages:
    #         st.chat_message(msg["role"]).write(msg["content"])
    #     if ("location" in st.session_state):
    #         if prompt:
    #             st.session_state.messages.append({"role": "user", "content": prompt})
    #             st.chat_message("user").write(prompt)
    #
    #             response = st.chat_message("assistant").write_stream(
    #                 get_llm_response(prompt, st.session_state.messages))
    #             st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
