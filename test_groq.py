import os
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Read API key and model name
api_key = os.getenv("GROQ_API_KEY")
model = os.getenv("GROQ_MODEL")

# Check whether API key exists
if not api_key:
    raise ValueError("Groq API key not found!")

# Create Groq client
client = Groq(api_key=api_key)

# Send a test message
response = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "user",
            "content": "Explain what an insurance claim is in two sentences."
        }
    ],
)

# Display the response
print(response.choices[0].message.content)