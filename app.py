import speech_recognition as sr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import AIMessage, HumanMessage
import os
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.agent_toolkits import GmailToolkit 
from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)
from langchain.agents import tool
from gmail import gmail_create_draft
from gmail import gmail_find
from gcalendar import gcalendar_create_event
from gcalendar import next_events
from gcalendar import find_event
from gdrive import drive_find_file
from gcontacts import find_contact
from gcontacts import all_contacts

load_dotenv() 

speech = sr.Recognizer()
import subprocess

def text_to_speech(command):
    subprocess.run(["say", "-v", "Daniel", command])
def understand_voice():
    voice_text = ""
    with sr.Microphone() as source:
        print("I'm listening ... ")
        audio = speech.listen(source, 5)
    try:
        voice_text = speech.recognize_google(audio, language='en')
    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        print('Network error.')
    print("You: " + voice_text)
    return voice_text

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-3.5-turbo"
    )

tools = [gmail_create_draft, gcalendar_create_event, next_events, find_event, gmail_find, drive_find_file, find_contact, all_contacts]

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from datetime import date

today = date.today()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You perform a variety of tasks using Google APIs."
            "Here are some things to keep in mind:"
            f"- Today is {today}"
            "Time zone is always Pacific."
            "- For gmail_find, the parameters 'after' and 'before' must be in the form 'yyyy/mm/dd'.",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

chat_history = []

llm_with_tools = llm.bind_tools(tools)

from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("Jarvis: Hello, how can I help you?")
text_to_speech("Hello, how can I help you?")
while True:
    user_input = understand_voice()
    if "thanks" in user_input.lower():
        break
    if "bye" in user_input.lower():
        break
    if "thank you" in user_input.lower():
        break
    result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
    chat_history.extend(
    [
        HumanMessage(content=user_input),
        AIMessage(content=result['output']),
    ]
    )
    text_to_speech(result['output'])