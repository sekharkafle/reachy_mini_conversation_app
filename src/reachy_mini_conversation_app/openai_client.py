from openai import OpenAI
from reachy_mini_conversation_app.config import config
import logging

prompt = '''You are a helpful assistant. You will be provided with the brief description of 
    what is going on in the surrounding and you will respond with appropriate things to do.
    Examples:
    if you see a person holding a remote, then suggest some classic TV shows or movies to watch.
    if you see a young person, suggest an activity to do or book to read
    if you see one or more individuals in kitchen with ingredients, suggest n appropriate recipe.
    if you see no human in the surroundings, just say "NO".

    Be brief in your response.
    '''

logger = logging.getLogger(__name__)
def get_llm_response(input_text: str) -> str:
    # Set the base_url to your local server's address
    client = OpenAI(
        base_url=config.LLM_URL, 
        api_key="sk-XXX" # API key is often not needed or can be a placeholder for local servers
    )

    completion = client.chat.completions.create(
        model="nemotron",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": input_text},
        ],
    )
    logger.info(f"LLM response: {completion.choices[0].message.content}")
    return completion.choices[0].message.content
