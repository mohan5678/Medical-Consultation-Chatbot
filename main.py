import streamlit as st
from typing import TypedDict, Annotated, Sequence, List, Literal, TypeVar, Dict
from langgraph.graph import START, END, StateGraph
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, BaseMessage
from langgraph.graph.message import add_messages
from langchain.prompts import PromptTemplate

# Assuming API_KEY is defined somewhere in your code
API_KEY = "YOUR_API_KEY"

model = ChatGroq(api_key=API_KEY, model="meta-llama/llama-4-scout-17b-16e-instruct")

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_node: str

def supervisor(state: AgentState) -> Dict:
    """Supervise and direct the workflow"""
    system_prompt = SystemMessage(content="""You are a medical supervisor.
    Your role is to orchestrate the flow of a medical consultation by selecting the most appropriate next agent based on the current conversation history.

    Choose EXACTLY ONE of the following options:
    - "query_enhancer_agent" - for correcting grammatical mistakes and enhancing the user's input for better model responses.
    - "symptom_analyzer_agent" - for analyzing patient symptoms and providing initial information.
    - "medicine_predictor_agent" - for recommending specific medications based on the identified symptoms.
    - "END" - if the consultation is complete and no further agent action is required.

    IMPORTANT:
        Respond with ONLY ONE of these exact terms from this ["symptom_analyzer_agent","medicine_predictor_agent","query_enhancer_agent","END"]. 
        Do not include any extra text, punctuation, or conversational filler.
        Do not select "END", without medications
    """)
    
    response = model.invoke([system_prompt] + state['messages'])
    response_message = AIMessage(
        content=response.content,
        name="supervisor"
    )
    
    state['messages'].append(response_message)
    
    decision = response.content.strip().lower()
    print(f"DEBUG - Decision after processing: '{decision}'")
    
    next_node_decision = ""
    if "end" in decision: 
        next_node_decision = "END"
    elif "symptom_analyzer_agent" in decision: 
        next_node_decision = "symptom_analyzer_agent"
    elif "medicine_predictor_agent" in decision: 
        next_node_decision = "medicine_predictor_agent"
    elif "query_enhancer_agent" in decision: 
        next_node_decision = "query_enhancer_agent"
    else:
        print(f"No match found, defaulting to symptom_analyzer")
        next_node_decision = "symptom_analyzer_agent"
        
    return {"messages": [response_message], "next_node": next_node_decision}

def symptom_analyzer(state: AgentState) -> Dict:
    """Analyze symptoms and provide potential conditions."""
    
    system_prompt = SystemMessage(content="""You are a medical expert analyzing patient-reported symptoms. Based on the patient's description, identify and categorize their symptoms and potential conditions.

    **Provide your analysis in the following strict format:**

    SYMPTOMS IDENTIFIED:
        - [List all clearly identified symptoms]
    POTENTIAL CONDITIONS:
        - [List possible medical conditions, avoiding definitive diagnosis]
    SEVERITY:
        - [Rate as Low/Medium/High, with brief justification]
    IMMEDIATE ACTION:
        - [Yes/No] - [Explain if immediate professional medical attention is recommended, or provide general advice for self-care if 'No']
    """)
    
    response = model.invoke([system_prompt] + state['messages'])
    response_message = AIMessage(
        content=response.content,
        name="symptom_analyzer"  
    )

    state['messages'].append(response_message)
    return {"messages": state['messages']}

def medicine_predictor(state: AgentState) -> Dict:
    """Predict appropriate medications based on conditions."""
    system_prompt = SystemMessage(content="""You are a medical expert whose sole purpose is to recommend over-the-counter medications based on the patient's analyzed symptoms and conditions provided.

    **Provide recommendations in the following STRICT format:**

    RECOMMENDED MEDICATIONS:
        - [Medication Name] [Dosage], [Frequency] for [Symptom/Condition]
        - [Additional medications if applicable]

    PRECAUTIONS:
        - [List specific precautions, e.g., "Consult a doctor if...", "Do not exceed..."]

    SIDE EFFECTS:
        - [List common and important potential side effects]
    """)
    
    response = model.invoke([system_prompt] + state['messages'])
    response_message = AIMessage(
        content=response.content,
        name="medicine_predictor"  
    )
    
    state['messages'].append(response_message)
    return {"messages": state['messages']}

# def validator(state: AgentState) -> Dict:
#     """Validate the medications based on the patient illness."""
#     system_prompt = SystemMessage(content="""You are a medical expert validating medication recommendations.
#     Review the patient's symptoms, conditions, and the proposed medications (including dosage, precautions, and side effects) provided by the previous agent.

#     **Your primary task is to ensure the recommendations are accurate and safe.**

#     **If you find any inaccuracies, omissions, or safety concerns:**
#     - **CORRECT** the specific piece of information.
#     - Then, output **ONLY the revised medication names and their dosages.**

#     IMPORTANT:
#     - Output **ONLY the medication names and their dosages not more than 3 lines** as originally provided by the previous agent.
#     - Don't just inform medication is validated and correct.
#     """
#     )
    
#     response = model.invoke([system_prompt] + state["messages"])
#     response_message = AIMessage(
#         content=response.content,
#         name="validator"  
#     )
    
#     state['messages'].append(response_message)
#     return {"messages": state['messages']}

def query_enhancer(state: AgentState) -> Dict:
    """Enhances the user query"""
    system_prompt = SystemMessage(content="""You are a **Query Enhancer**. 
    Your role is to **correct grammatical mistakes** in the user's query and **enhance it for a better response** from the model.
    """
    )
    
    response = model.invoke([system_prompt] + state["messages"])
    response_message = AIMessage(
        content=response.content,
        name="query_enhancer"  
    )
    
    state['messages'].append(response_message)
    return {"messages": state['messages']}


# Creating the workflow
work_flow = StateGraph(AgentState)
work_flow.add_node("supervisor", supervisor)
work_flow.add_node("query_enhancer_agent", query_enhancer)
work_flow.add_node("symptom_analyzer_agent", symptom_analyzer)
work_flow.add_node("medicine_predictor_agent", medicine_predictor)

work_flow.set_entry_point("supervisor")

work_flow.add_conditional_edges(
    "supervisor",
    lambda state: state["next_node"], 
    {
        "query_enhancer_agent":"query_enhancer_agent",
        "symptom_analyzer_agent": "symptom_analyzer_agent",
        "medicine_predictor_agent": "medicine_predictor_agent",
        "END": END, 
    }
)

members = ["symptom_analyzer_agent", "medicine_predictor_agent", "query_enhancer_agent"]

for member in members:
    work_flow.add_edge(member, "supervisor")

app = work_flow.compile()

def main():
    st.title("Medical Consultation Chatbot")

    user_input = st.text_area("Describe your symptoms:", height=100)

    if st.button("Get Medication Recommendations"):
        if user_input:
            query = [HumanMessage(content=user_input)]
            response = app.invoke({"messages": query}, {"recursion_limit": 20})

            # Find and display medicine_predictor message
            medicine_predictor_output = None
            for msg in response['messages']:
                if hasattr(msg, 'name') and msg.name == 'medicine_predictor':
                    medicine_predictor_output = msg.content
                    break
            if medicine_predictor_output is not None:
                st.write(medicine_predictor_output)

if __name__ == "__main__":
    main()