from datetime import datetime
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

@tool
def gcalendar_create_event(name: str, attendee_id: str, attendee_email: str, start: str, end: str):
    """Create a google calendar event.
    """
    credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/", "https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/contacts"],
        client_secrets_file="credentials.json",
    )

    event = {
  'summary': name,
  'start': {
    'dateTime': start,
    'timeZone': "America/Los_Angeles",
  },
  'end': {
    'dateTime': end,
    'timeZone': "America/Los_Angeles",
  }
    }

    service = build("calendar", "v3", credentials=credentials)
    event = service.events().insert(calendarId='stavgao@gmail.com', body=event).execute()

@tool
def next_events(number: int):
  """Prints the start and name of a specific number of events on the user's calendar.
  """
  credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/", "https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/contacts"],
        client_secrets_file="credentials.json",
    )

  try:
    service = build("calendar", "v3", credentials=credentials)

    now = datetime.datetime(2024, 7, 31).isoformat() + "Z" 
    events_result = (
        service.events()
        .list(
            calendarId="primary",
            timeMin=now,
            maxResults=number,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])

    if not events:
      print("No upcoming events found.")
      return

    tot = ""
    for event in events:
      start = event["start"].get("dateTime", event["start"].get("date"))
      tot += "Summary: " + event["summary"] + " " + "Start Time: " + start + "/n"
    return tot

  except HttpError as error:
    print(f"An error occurred: {error}")

import requests

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
    "You are an assistant that helps users locate events."
    "Based on the description the user gives you, use keywords to locate the event."
    "Return the summary of the event as well as the specific time it occurred."
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
def find_event(description: str, start_year: int, start_month: int, start_day: int, end_year: int, end_month: int, end_day: int):
  """Finds a specific event on user's calendar based on their description of it and when it occurs/occurred.
  """
  credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/", "https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/contacts"],
        client_secrets_file="credentials.json",
    )
  after = datetime(start_year, start_month, start_day).isoformat() + 'Z'
  before = datetime(end_year, end_month, end_day).isoformat() + 'Z'
  try:
    access_token = credentials.token
    list_url = f'https://www.googleapis.com/calendar/v3/calendars/primary/events'\
                f'?timeMin={after}' \
                f'&timeMax={before}' \

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json'
    }

    response = requests.get(list_url, headers=headers)
    
    tot = ""
    if response.status_code == 200:
      events = response.json().get('items', [])
      if events:
        for event in events:
          tot += f"Event Summary: {event.get('summary')}"
          tot += (f" Start Time: {event.get('start', {}).get('dateTime')}")
          tot += (f" End Time: {event.get('end', {}).get('dateTime')}")
          tot += '\n'
      else:
          print("No events found.")
    else:
      print(f"Failed to retrieve events. Status code: {response.status_code}")
    filename = "events.txt"
    with open(filename, "w") as file:
        file.write(tot)
    loader = TextLoader("events.txt")

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
  except HttpError as error:
    print(f"An error occurred: {error}")