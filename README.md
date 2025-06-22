# Medical Consultation Chatbot with LangGraph

## Project Overview
This project implements a multi-agent conversational chatbot designed to assist with initial medical consultations. Built using LangGraph for orchestrating complex agentic workflows and powered by Groq for high-performance LLM inference, it provides a Streamlit-based user interface for easy interaction.

The chatbot leverages a supervisor agent to dynamically route user queries through specialized agents for query enhancement, symptom analysis, and medication prediction.

## Features
**Query Enhancement**: Automatically corrects grammatical mistakes and enhances user-provided symptom descriptions for better understanding by subsequent agents.
**Symptom Analysis**: Identifies and categorizes reported symptoms, suggests potential (non-diagnostic) conditions, assesses severity, and recommends immediate actions.
**Medicine Prediction**: Recommends over-the-counter medications based on analyzed symptoms, including dosage, precautions, and potential side effects.
**Intelligent Orchestration**: A central supervisor agent directs the conversation flow, ensuring the most relevant agent processes each stage of the consultation.
**Streamlit UI**: Provides a user-friendly web interface for inputting symptoms and viewing recommendations.

## Architecture & Workflow
The chatbots core logic is built with LangGraph, which allows for defining agent interactions as a state graph:

1.  **supervisor**: The entry point and central orchestrator. It analyzes the entire conversation history and decides which specialized agent should handle the next turn:
    * query_enhancer_agent
    * symptom_analyzer_agent
    * medicine_predictor_agent
    * END (when the consultation is complete)
2.  **query_enhancer_agent**: Focuses on refining the users raw input. It corrects grammar and rephrases the query for optimal clarity before it proceeds to medical analysis.
3.  **symptom_analyzer_agent**: Receives the (potentially enhanced) query and provides a structured analysis of symptoms, potential conditions, severity, and immediate actions.
4.  **medicine_predictor_agent**: Based on the analyzed symptoms and conditions (from the symptom_analyzer), this agent suggests over-the-counter medications, including usage details and warnings.
5.  **Loop Back to Supervisor**: After an agent completes its task, the flow returns to the supervisor to decide the next logical step, ensuring a cohesive multi-turn conversation.

## Setup
Follow these steps to get the project up and running on your local machine.

### Prerequisites
* Python 3.9+
* A Groq API Key
* Git (optional, for cloning the repository)

### 1. Clone the Repository (Optional, if you have the files already)
bash
git clone <your-repository-url>
cd <your-repository-name> # e.g., medical_chatbot
