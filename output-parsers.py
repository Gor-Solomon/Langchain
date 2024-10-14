from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)


class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(description="The age of the person")
    address: dict = Field(description="The address of the persons")


def call_string_output_parser():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tell me a jock about the following subject."),
        ("human", "{input}")
    ])

    parser = StrOutputParser()
    chain = prompt | model | parser

    return chain.invoke({
        "input": "dog"
    })


def call_list_output_parser():
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Generate a list of 10 synonyms for the following word. Return the results as comma separated & "
                       "new line."),
            ("human", "{input}")
        ]
    )

    parser = CommaSeparatedListOutputParser()

    chain = prompt | model | parser

    return chain.invoke({"input": "happy"})


def call_json_output_parser():
    # Define the prompt with format instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract the information from the following phrase and return a valid JSON object matching the schema: {format_instructions}. Only output the JSON object."),
        ("human", "{phrase}")
    ])

    # Create a JSON output parser that maps to the Person class
    parser = JsonOutputParser(pydantic_object=Person)

    # Chain the prompt, model, and parser together
    chain = prompt | model | parser

    # Invoke the chain with the input phrase
    return chain.invoke({
        "phrase": "Maiximilus is 39 years old who has lived in 9th alley of Stockholm and 25B London and Buckingham",
        "format_instructions": parser.get_format_instructions()
    })


result = call_json_output_parser()
print(result)
