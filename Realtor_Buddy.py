import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_oauth import OAuth2Component
import json
import base64
import os
from dotenv import load_dotenv
load_dotenv()

# TODO: Uncomment once the SSO issue is fixed
CLIENT_ID = os.environ.get("SSO_CLIENT_ID")
CLIENT_SECRET = os.environ.get("SSO_CLIENT_SECRET")
AUTHORIZE_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"
REDIRECT_URI = os.environ.get("OAUTH_REDIRECT_URI")

def authentication():
    if "auth" not in st.session_state:
        print(""""auth" not in st.session_state""")
        # create a button to start the OAuth2 flow
        oauth2 = OAuth2Component(CLIENT_ID, CLIENT_SECRET, AUTHORIZE_ENDPOINT, TOKEN_ENDPOINT, TOKEN_ENDPOINT,
                                 REVOKE_ENDPOINT)
        result = oauth2.authorize_button(
            name="Continue with Google",
            icon="https://www.google.com.tw/favicon.ico",
            # TODO: Read from env file the redirect uri
            redirect_uri=REDIRECT_URI,
            scope="openid email profile",
            key="google",
            extras_params={"prompt": "consent", "access_type": "offline"},
            use_container_width=True,
            pkce='S256',
        )

        if result:
            print(result)
            st.write(result)
            # decode the id_token jwt and get the user's email address
            id_token = result["token"]["id_token"]
            # verify the signature is an optional step for security
            payload = id_token.split(".")[1]
            # add padding to the payload if needed
            payload += "=" * (-len(payload) % 4)
            payload = json.loads(base64.b64decode(payload))
            email = payload["email"]
            print(email)

            st.session_state["auth"] = email
            st.session_state["token"] = result["token"]
            st.rerun()
    else:
        print(""""auth" in st.session_state""")
        st.write("You are logged in!")
        st.write(st.session_state["auth"])

        email = st.session_state["auth"]
        # get_user_details_and_show_specific_data(email)

        # st.write(st.session_state["token"])
        if st.button("Logout"):
            del st.session_state["auth"]
            del st.session_state["token"]
            st.rerun()

def clear_chat_session():
    # Delete all the items necessary in Session state
    for key in st.session_state:
        if key=='chat_history' or key == 'messages':
            del st.session_state[key]

if __name__ == "__main__":
    st.set_page_config(
        page_title="Realtor Buddy",
        page_icon="🏡",
    )

    authentication()
    st.write("# Welcome to Realtor Buddy! 🏡")
    st.write("Chat with Realtor Buddy to find out your dream house!")

    if st.button("Text Chat"):
        clear_chat_session()
        switch_page("Text Chat")
    if st.button("Multimodal Chat"):
        clear_chat_session()
        switch_page("Multimodal Chat")
