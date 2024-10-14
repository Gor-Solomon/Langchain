from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

# Instantiate Model
llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate a list of 10 synonyms for the following word. Return the results as comma separated & "
                   "new line."),
        ("human", "{input}")
    ]
)

# Create LLM Chain
chain = prompt | llm

response = chain.invoke({"input": "unpleased"})
print(response.content)
