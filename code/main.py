import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.1,
    max_tokens=512,
    timeout=30,
    max_retries=3,
    api_key=os.getenv("GOOGLE_API_KEY"),
)

messages = [
  SystemMessage(content={}),
  HumanMessage(content="""
    I found an SQL injection vulnerability in a login form during a penetration test. 
    What steps should I take to remediate this issue and prevent similar attacks in the future?
  """)
]
  
response = llm.invoke(messages)
print(response.content)