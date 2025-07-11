from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda


load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.8)

template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can answer questions about the user's input."),
        ("user", "can you tell me features of asus laptops?")
    ]
)

def pro_features(features: str):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a expert of reviewing products. You are given a list of features of a product and you need to review the product and give the pros of the product."),
            ("user", "can you tell me pros of {features}")
        ]
    )
    return pros_template.invoke({"features": features})

def cons_features(features: str):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a expert of reviewing products. You are given a list of features of a product and you need to review the product and give the cons of the product."),
            ("user", "can you tell me cons of {features}")
        ]
    )    
    return cons_template.invoke({"features": features})

pros_chain = (RunnableLambda(pro_features) | llm | StrOutputParser())
cons_chain = (RunnableLambda(cons_features) | llm | StrOutputParser())
    
chain = template | llm | StrOutputParser() | RunnableParallel(branches={"pros": pros_chain, "cons": cons_chain}) | RunnableLambda(lambda x: x["branches"]["pros"] + "\n" + x["branches"]["cons"])

print(chain.invoke({}))