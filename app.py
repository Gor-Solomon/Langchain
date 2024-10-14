import os
from langchain_openai import ChatOpenAI

openAi_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(api_key=openAi_key,
                 model="gpt-3.5-turbo",
                 temperature=0.9,
                 max_tokens=1000)
text = "Write a poem about AI..."
results = llm.invoke(text)

print(results.content)

