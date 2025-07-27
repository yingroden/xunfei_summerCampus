# /// script
# requires-python = ">=3.12"
# dependencies = ["langchain_openai","openai","pydantic"]
# ///

from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from pydantic import SecretStr
import os
from openai import OpenAI

MODEL_CONFIGS = {
    "ollama":{
        "base_url":"http://localhost:11434/v1",
        "api_key":"ollama",
        "chat_model":"deepseek-r1:1.5b",
    },
    "deepseek":{
        "base_url":"https://api.deepseek.com",
        "api_key":"sk-6eefc810cffc464782115a242d356b7f",
        "chat_model":"deepseek-chat",
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": "sk-219b35d688eb40e2aeec7dd0aae84bda",
        "chat_model": "qwen-max",
    }
}

DEFAULT_MODEL = "ollama"
DEFAULT_TEMPERATURE = 0.7


def init_model(llm_type) -> ChatOpenAI:
    config = MODEL_CONFIGS[llm_type]
    if llm_type == "ollama":
        os.environ["OPENAI_API_KEY"] = "NA"


    return ChatOpenAI(
        model=config["chat_model"],
        base_url=config["base_url"],
        api_key=SecretStr(config["api_key"]),
        temperature=DEFAULT_TEMPERATURE,
        timeout=10,
        max_retries=3,
    )
    
    

def test_model(llm_type):

    config = MODEL_CONFIGS[llm_type]

    if llm_type == "ollama":
        os.environ["OPENAI_API_KEY"] = "NA"

    llm_chat = init_model(llm_type)
    response = llm_chat.invoke("Hello, how are you?")
    #   流式输出
    for chunk in llm_chat.stream("介绍一下你自己?"):
        print(chunk.content, end="", flush=True)
    return response.content 

if __name__ == "__main__":
    # print(test_model("deepseek"))
    test_model("qwen")