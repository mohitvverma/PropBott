from PropBots.constants import Config
from PropBots.CustomPropException import PropBotException
from PropBots.logger import logging
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
import os
import base64
import sys


image_elements = []
image_summaries = []

# Load environment variables
load_dotenv()


class PropBotModels:
    def __init__(self):
        try:
            self.api_key = os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise PropBotException("OpenAI API key not found in environment variables.", sys)
            self.max_tokens = Config.MAX_TOKENS_VISION
            logging.info("PropBotModels initialized successfully.")
        except Exception as e:
            logging.error(f"Error during PropBotModels initialization: {str(e)}")
            raise PropBotException("Failed to initialize PropBotModels", sys) from e

    @staticmethod
    def encode_image(image_path):
        """
        Encodes the image to base64.

        Args:
            image_path (str): Path to the image to be encoded.

        Returns:
            str: The base64-encoded image.
        """
        try:
            with open(image_path, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')
                logging.info(f"Image at {image_path} encoded successfully.")
                return encoded_image
        except FileNotFoundError as e:
            logging.error(f"Image file not found: {image_path}")
            raise PropBotException(f"Image file not found: {image_path}", sys) from e
        except Exception as e:
            logging.error(f"Error encoding image {image_path}: {str(e)}")
            raise PropBotException(f"Failed to encode image: {image_path}", sys) from e

    def summarize_image(self, encoded_image):
        """
        Sends an encoded image to the OpenAI model for summarization.

        Args:
            encoded_image (str): Base64-encoded image.

        Returns:
            str: Summary of the image contents.
        """
        try:
            prompt = [
                SystemMessage(content=Config.IMAGE_SYSTEM_MESSAGE),
                HumanMessage(content=[
                    {
                        "type": "text",
                        "text": "Describe the contents of this image and do not miss any important details"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        },
                    },
                ])
            ]
            logging.info("Sending image summarization request to the model.")
            response = ChatOpenAI(model=Config.VISION_MODEL_NAME,
                                  openai_api_key=self.api_key,
                                  max_tokens=self.max_tokens).invoke(prompt)
            logging.info("Image summarized successfully.")

            return response.content

        except Exception as e:
            logging.error(f"Error summarizing image: {str(e)}")
            raise PropBotException("Failed to summarize image", sys) from e

    @staticmethod
    def chat_model():
        """
        Initializes and returns a chat model.

        Returns:
            ChatOpenAI: Chat model instance.
        """
        try:
            llm = ChatOpenAI(temperature=Config.TEMPERATURE, model=Config.GPT_MODEL_NAME)
            logging.info("Chat model initialized successfully.")
            return llm
        except Exception as e:
            logging.error(f"Error initializing chat model: {str(e)}")
            raise PropBotException("Failed to initialize chat model", sys) from e


    def ai21_chat_model(self):
        """
        Initializes and returns a chat model.

        Returns:
        ChatOpenAI: Chat model instance.
        """
        try:
            llm = ChatOpenAI(temperature=Config.TEMPERATURE, model=Config.GPT_MODEL_NAME)
            logging.info("Chat model initialized successfully.")
            return llm

        except Exception as e:
            logging.error(f"Error initializing chat model: {str(e)}")
            raise PropBotException("Failed to initialize chat model", sys) from e

