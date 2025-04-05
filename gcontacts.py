import os.path

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
    "You are an assistant that helps users locate people."
    "Based on a slightly accurate transcription of the user saying their name, you must find the person out of a list of names."
    "Always return the name and email. Never ask for clarification."
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
def find_contact(description: str):
  """Figures out the specific contact the user is looking for from the name. Returns name and email address.
  """

  credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/", "https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/contacts"],
        client_secrets_file="credentials.json",
    )

  try:
    service = build("people", "v1", credentials=credentials)

    results = (
        service.people()
        .connections()
        .list(
            resourceName="people/me",
            personFields="names,emailAddresses",
        )
        .execute()
    )
    connections = results.get("connections", [])
    people = ""
    for person in connections:
      names = person.get("names", [])
      email_addresses = person.get("emailAddresses", [])
      name = names[0].get("displayName")
      email = email_addresses[0].get("value")
      people += ("Name: " + name + " ")
      people += ("Email: " + email + "\n")
    filename = "contacts.txt"
    with open(filename, "w") as file:
        file.write(people)
    loader = TextLoader("contacts.txt")

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

  except HttpError as err:
    print(err)

@tool
def all_contacts():
  """Returns all google contacts the user has.
  """

  credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/", "https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/contacts"],
        client_secrets_file="credentials.json",
    )

  try:
    service = build("people", "v1", credentials=credentials)

    results = (
        service.people()
        .connections()
        .list(
            resourceName="people/me",
            personFields="names,emailAddresses",
        )
        .execute()
    )
    connections = results.get("connections", [])
    people = ""
    for person in connections:
      names = person.get("names", [])
      email_addresses = person.get("emailAddresses", [])
      name = names[0].get("displayName")
      email = email_addresses[0].get("value")
      people += ("Name: " + name + " ")
      people += ("Email: " + email + "\n")
    return people

  except HttpError as err:
    print(err)