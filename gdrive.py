import google.auth
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)

from datetime import datetime

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
from langchain.agents import tool

llm = ChatOpenAI(model="gpt-3.5-turbo")

system_prompt = (
    "You are an assistant that helps users locate google drive files."
    "Based on the description the user gives you, use keywords to locate the file."
    "Return the title of the file."
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
def drive_find_file(description: str, start_year: int, start_month: int, start_day: int, end_year: int, end_month: int, end_day: int):
  """Search file in drive location
  """
  credentials = get_gmail_credentials(
        token_file="token.json",
        scopes=["https://mail.google.com/", "https://www.googleapis.com/auth/calendar", "https://www.googleapis.com/auth/drive", "https://www.googleapis.com/auth/contacts"],
        client_secrets_file="credentials.json",
    )
  tot = ""
  service = build("drive", "v3", credentials=credentials)
  files = []
  page_token = None
  start_date = datetime(start_year, start_month, start_day).isoformat() + 'Z'
  end_date = datetime(end_year, end_month, end_day).isoformat() + 'Z'
  while True:
    query = f"modifiedTime >= '{start_date}' and modifiedTime <= '{end_date}'"
    response = (
        service.files()
        .list(
            q=query,
            spaces="drive",
            fields="nextPageToken, files(id, name, modifiedTime)"
        )
        .execute()
    )
    for file in response.get("files", []):
      tot += f'Found file: {file.get("name")}, {file.get("id")}'
    files.extend(response.get("files", []))
    page_token = response.get("nextPageToken", None)
    if page_token is None:
      break
  filename = "drive.txt"
  with open(filename, "w") as file:
      file.write(tot)
  loader = TextLoader("drive.txt")

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