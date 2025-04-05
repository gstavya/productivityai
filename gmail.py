import base64
from email.message import EmailMessage

import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain.agents import tool
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)

@tool
def gmail_create_draft(thread_id: str, send: bool, to: str, subject: str, content: str):
    """Create and insert a draft email or send it.
    Print the returned draft or sent message's ID.
    Returns: Draft or Message object, including draft/message ID and meta data.

    Load pre-authorized user credentials from the environment.
    """
    credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/", "https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/contacts"],
        client_secrets_file="credentials.json",
    )

    try:
        service = build("gmail", "v1", credentials=credentials)

        message = EmailMessage()
        message.set_content(content)
        message["To"] = to
        message["From"] = "stavgao@gmail.com"
        message["Subject"] = subject

        encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()

        create_message = {"raw": encoded_message}

        if thread_id:
            create_message['threadId'] = thread_id
        
        if not send:
            draft = service.users().drafts().create(userId="me", body={"message": create_message}).execute()
            print(f'Draft id: {draft["id"]}\nDraft message: {draft["message"]}')
        else:
            sent_message = service.users().messages().send(userId="me", body=create_message).execute()
            print(f'Message id: {sent_message["id"]}\nMessage: {sent_message}')

    except HttpError as error:
        print(f"An error occurred: {error}")
        return None

    return draft if not send else sent_message

if __name__ == "__main__":
    gmail_create_draft()

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langchain.agents import tool
import requests
from langchain_core.prompts import ChatPromptTemplate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from langchain_community.document_loaders import TextLoader
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

system_prompt = (
    "You are an assistant that helps users locate emails."
    "Based on the description the user gives you, use keywords to locate the email."
    "Return the subject of the email, the thread ID, and the link."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

@tool
def gmail_find(description: str, sender: str, after: str, before: str):
    """Returns the email that the user is looking for.
    """
    credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/", "https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/contacts"],
        client_secrets_file="credentials.json",
    )

    access_token = credentials.token
    list_url = f'https://www.googleapis.com/gmail/v1/users/me/messages?q=in:inbox after:{after} before:{before}'

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    response = requests.get(list_url, headers=headers)

    tot = ""
    result = ""
    if response.status_code == 200:
        messages = response.json().get('messages', [])
        
        for message in messages:
            message_id = message['id']
            thread_id = message['threadId']
            message_url = f'https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}'
            msg_response = requests.get(message_url, headers=headers)
            if msg_response.status_code == 200:
                msg_details = msg_response.json()
                snippet = msg_details.get('snippet', 'No snippet available')
                subject = 'No subject available'
                headers_list = msg_details.get('payload', {}).get('headers', [])
                
                for header in headers_list:
                    if isinstance(header, dict) and header.get('name') == 'Subject':
                        subject = header.get('value')
                    if header.get('name') == 'From':
                        sender = header.get('value')

                body = 'No body available'
                payload = msg_details.get('payload', {})
                parts = payload.get('parts', [])
            
                for part in parts:
                    if part.get('mimeType') == 'text/plain':
                        body = part.get('body', {}).get('data', 'No body available')
                    elif part.get('mimeType') == 'text/html':
                        body = part.get('body', {}).get('data', 'No body available')

                if body != 'No body available':
                    import base64
                    body = base64.urlsafe_b64decode(body).decode('utf-8')
                tot += "Subject: " + subject + " Snippet: " + snippet + "Sender: " + sender + "Thread ID: " + thread_id + "Message Link: " + message_url + "/n"
    filename = "inbox.txt"
    with open(filename, "w") as file:
        file.write(tot)
    loader = TextLoader("inbox.txt")

    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(document)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    retriever = vectorstore.as_retriever()

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    results = rag_chain.invoke({"input": description})
    result = results['answer']
    return result
