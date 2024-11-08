from langchain_openai import ChatOpenAI

from src.utils import get_env_variable

def gpt_4o_mini(prompt):
    OPENAI_API_KEY = get_env_variable("OPENAI_API_KEY")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    return llm.invoke(prompt).content.strip()