# import openai
# from langchain_openai import ChatOpenAI
# openai_secret = os.environ['OPENAI_API_KEY']
import langchain
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains import SimpleSequentialChain, LLMChain
from langchain.agents import AgentType, initialize_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.memory import ConversationBufferMemory

gemini_secret = os.environ['GEMINI_API_KEY']

# Initializing the LLM
llm = ChatGoogleGenerativeAI(
    # model="gemini-1.5-pro",
    model="gemini-2.0-flash",
    temperature=0.3,
    max_tokens=256,
    timeout=None,
    max_retries=2,
    api_key=gemini_secret,
    # other params...
)


def simple_call(prompt):
  response = llm.invoke(prompt)
  return response.content


def call_with_template(year):
  # Define the prompt (Can change)
  prompt = ChatPromptTemplate.from_messages([
      ("system",
       "You are a history expert who provides major historical events for a given year."
       ),
      ("human",
       "Tell me a significant historical event that happened in the year {year}."
       ),
  ])

  # Chain the prompt with the LLM
  chain = prompt | llm

  # Invoke the chain with a specific year
  response = chain.invoke({"year": year})

  # Print the response
  return response.content


# Using sequential chain
def ecommerce_chain(llm):
  # First prompt: Find an e-commerce store from a product name
  store_prompt = PromptTemplate(
      input_variables=["product"],
      template=
      "Which e-commerce store is best for buying {product}? Only list a single store."
  )

  # Second prompt: Get popular products from the store name
  products_prompt = PromptTemplate(
      input_variables=["store"],
      template=
      "List some popular products available on {store}, separated by commas. Only list the products."
  )

  # First chain: Gets the store name
  store_chain = LLMChain(llm=llm, prompt=store_prompt)

  # Second chain: Gets the products from that store
  products_chain = LLMChain(llm=llm, prompt=products_prompt)

  # Create a sequential chain
  overall_chain = SimpleSequentialChain(chains=[store_chain, products_chain],
                                        verbose=True)

  return overall_chain


# # Initialize the chain
# chain = ecommerce_chain(llm)

# # Invoke the chain
# result = chain.invoke("smartphone")
# print(result)

# -----------------------------------------------#


# Using an agent as specified by the tools
def search_agent(llm, prompt):
  tools = load_tools(["wikipedia", "llm-math"], llm=llm)
  agent = initialize_agent(tools,
                           llm,
                           agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                           verbose=True)
  output = agent.run(prompt)
  return output


# print(search_agent(llm, "What is the release history of DeepSeek R1?"))

# -----------------------------------------------#
# Using memory
# Initialize Memory
memory = ConversationBufferMemory(memory_key="chat_history",
                                  return_messages=True)

# Define Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an intelligent assistant who remembers past conversations."),
    ("human", "{chat_history} \nUser: {input}"),
])

#  Create LLM Chain with Memory
conversation_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)


#  Function to Interact with Memory-Enabled Chain
def chat_with_memory(user_input):
  response = conversation_chain.invoke({"input": user_input})
  return response["text"]  # Extract text from response


#  Example Usage
# print(chat_with_memory("Hello, remember the number 42."))
# print(chat_with_memory("Hey, who are you?"))
# print(chat_with_memory("What number I had asked to remember?"))
