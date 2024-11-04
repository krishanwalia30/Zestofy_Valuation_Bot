import time
import streamlit as st
import dspy
import google.generativeai as genai
import os

# Define the ValuationChatbot class that inherits from dspy.Signature
class MarketingChatbot(dspy.Signature):
  """You are a Marketing Chatbot, whose main aim is to answer marketing queries of the user.

  You may also be given the history of the prompts and responses, use this history as the context while answering the query.

  All your responses should be strictly specific to marketing domain.
  
  If anything not related to marketing is given in the query, you have to politely refuse to answer.
  """
  # If any query is not related to marketing, you should politely refuse to answer.
  history = dspy.InputField(desc="The history of prompts and responses")
  query = dspy.InputField(desc="The query of the user.")
  answer = dspy.OutputField(desc="The answer to the user's query.")

genai.configure(api_key=st.secrets['GOOGLE_API_KEY'])
gemini = dspy.Google(model='gemini-1.5-flash', api_key=os.environ["GOOGLE_API_KEY"], temperature=0.3)
dspy.settings.configure(lm=gemini)


# Define the CoT class that inherits from dspy.Module
class CoT(dspy.Module):
  def __init__(self):
    super().__init__()
    self.program = dspy.ChainOfThought(MarketingChatbot)
  
  def forward(self, history, query):
    return self.program(history=history, query=query)


# Initialize CoT module
chatbot = CoT()

# Response Stream Generator
def response_stream_generator(response_text:str):
   for word in (response_text).split(" "):
      yield word + " "
      time.sleep(0.1)

# Display app title
st.title(":blue[Ze]:green[st]:red[o]:orange[fy]  _Valuation Bot_ ")
st.text('Start by typing a message')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Write your message..."):
    # Storing the query
    query_from_user = prompt 
    history_of_user_interaction = str(st.session_state.messages)

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # TODO: Add to read the document here.
    answer_from_model = chatbot.forward(" ".join(history_of_user_interaction), query_from_user)

    response = f"Zestofy: \t {answer_from_model.answer}"

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.write_stream(response_stream_generator(response))
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    print("THE MESSAGE HISTORY IS: ", st.session_state.messages)


# File uploader in the sidebar
uploaded_files = st.sidebar.file_uploader(
    label="Upload any relevant document", type=["pdf"], accept_multiple_files=True
)
