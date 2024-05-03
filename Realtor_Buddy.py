import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title="Realtor Buddy",
    page_icon="🏡",
)

def clear_chat_session():
    # Delete all the items necessary in Session state
    for key in st.session_state:
        if key=='chat_history' or key == 'messages':
            del st.session_state[key]

st.write("# Welcome to Realtor Buddy! 🏡")
st.write("Chat with Realtor Buddy to find out your dream house!")

if st.button("Text Chat"):
    clear_chat_session()
    switch_page("Text Chat")
if st.button("Multimodal Chat"):
    clear_chat_session()
    switch_page("Multimodal Chat")
