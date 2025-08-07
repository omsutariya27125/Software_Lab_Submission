from langchain_groq import ChatGroq
from langchain.schema import HumanMessage
from dotenv import load_dotenv
import re

load_dotenv()
# Initialize ChatOpenAI client (adjust model and temperature as needed)
chat = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0)

CATEGORY_PROMPTS = {
    "order_issue": """
You are a helpful customer support agent.
The customer has an order issue.
Please empathize, ask for any missing details politely, and provide
next steps to resolve the order issue.
Customer message: {message}
dont sign the messages with anything other than - customer service bot
""",
    "refund_request": """
You are a helpful customer support agent.
The customer is requesting a refund.
Politely confirm the refund policy, ask for order details if needed,
and inform about the refund process.
dont sign the messages with anything other than - customer service bot
Customer message: {message}
""",
    "product_inquiry": """
You are a knowledgeable product expert.
The customer has a question about a product.
Provide clear, informative, and friendly answers.
dont sign the messages with anything other than - customer service bot
Customer message: {message}
""",
    "general_feedback": """
You are a customer service agent.
The customer is giving general feedback.
Thank them sincerely, acknowledge their feedback, and assure them it
will be reviewed.
dont sign the messages with anything other than - customer service bot
Customer message: {message}
""",
}

def classify_message_ai(message: str) -> str:
    classification_prompt = f"""
You are a helpful assistant that categorizes customer messages into
one of these categories only:
- order_issue
- refund_request
- product_inquiry
- general_feedback

Given the customer message below, reply with **only the category
name** that best fits the message.

Customer message: "{message}"
Category:
"""
    response = chat.invoke([HumanMessage(content=classification_prompt)])
    cleaned_response = re.sub(r'<think>.*?</think>', '',
                             response.content, flags=re.DOTALL).strip().lower()
    category = cleaned_response.split()[0] if cleaned_response else "general_feedback"
    if category not in CATEGORY_PROMPTS:
        category = "general_feedback"
    return category

def generate_response(category: str, message: str) -> str:
    prompt_template = CATEGORY_PROMPTS[category]
    prompt = prompt_template.format(message=message)

    response = chat.invoke([HumanMessage(content=prompt)])
    response_text = response.content

    if '</think>' in response_text:
        final_reply = response_text.split('</think>', 1)[1].strip()
    else:
        final_reply = response_text.strip()

    return final_reply


def main():
    print("Welcome to the Customer Support Auto-Responder. Type 'exit'to quit.")
    while True:
        user_message = input("\nCustomer message: ")
        if user_message.lower() == "exit":
            print("Goodbye!")
            break

        category = classify_message_ai(user_message)
        print(f"Category detected: {category}")

        reply = generate_response(category, user_message)
        print(f"Auto-Responder reply:\n{reply}")

if __name__ == "__main__":
    main()