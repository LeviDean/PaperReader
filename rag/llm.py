from typing import List, Dict
from openai import AzureOpenAI
from utils.config import Config

class LLMChat:
    def __init__(self, config: Config):
        """
        Initialize the LLMChat with a configuration.

        :param config: Configuration dictionary containing LLM settings.
        """
        self.config = config
        self.llm_client = AzureOpenAI(azure_endpoint=config.chat.endpoint, api_version=config.chat.api_version)
        
    def chat(self, messages: List[Dict], stream: bool = False):
        """
        Chat with the specified LLM.

        :param messages: List of message dictionaries to send.
        :return: The response from the LLM.
        """
        completion = self.llm_client.chat.completions.create(
            model=self.config.chat.deployment,
            messages=messages,
            temperature=1e-19,
            stream=stream,
        )
        if stream:
            return completion
        else:
            return completion.choices[0].message.content
        
if __name__ == "__main__":
    config = Config("./dev_config.yaml")
    llm = LLMChat(config)
    response = llm.chat([{"role": "user", "content": "Hello, how are you?"}], stream=True)
    
    response_stream = ""
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
            response_stream += chunk.choices[0].delta.content
            print(chunk.choices[0].delta.content, end="", flush=True)
    print("\n")
    print(response_stream)
