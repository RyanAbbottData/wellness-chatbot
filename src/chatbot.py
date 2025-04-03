import boto3
import json
import os
from dotenv import load_dotenv

from botocore.exceptions import ClientError


load_dotenv()


def get_query_from_prompt(query: str) -> str:
    """
    Takes a question, submitted in human speak by a user,
    and converts it into a SQL query.

    Parameters:
    -----------
    query (str):
        The question from the user

    Returns:
    -----------
    response_text (str):
        The SQL query
    """
    # Entrypoint to AWS bedrock
    client = boto3.client("bedrock-runtime",
                          aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
                          aws_secret_access_key=os.getenv("AWS_SECRET_KEY"),
                          region_name="us-east-1")

    # Set the model ID, e.g., Titan Text Premier.
    model_id = "amazon.titan-text-lite-v1"

    # Define the prompt for the model.
    prompt = f"""We have a sql database, where we have the following tables and columns:
                'diet', where 'date' and 'calories' are the columns;
                'fitness', where 'date' and calories' are the columns;
                'symptoms', where 'date' and 'symptom' are the columns. 
                Build a SQL query out of the following: {query}"""

    # Format the request payload using the model's native structure.
    native_request = {
        "inputText": prompt,
        "textGenerationConfig": {
            "maxTokenCount": 512,
            "temperature": 0.1,
        },
    }

    # Convert the native request to JSON.
    request = json.dumps(native_request)

    try:
        # Invoke the model with the request.
        response = client.invoke_model(modelId=model_id, body=request)

    except (ClientError, Exception) as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        exit(1)


    # Decode the response body.
    model_response = json.loads(response["body"].read())

    # Extract and print the response text.
    response_text = model_response["results"][0]["outputText"]

    return response_text
