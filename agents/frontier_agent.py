# # imports

import os
import re
import math
import json
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import chromadb
from items import Item
from testing import Tester
from agents.agent import Agent
import time

class FrontierAgent(Agent):

    name = "Frontier Agent"
    color = Agent.BLUE

    MODEL = "gpt-4o-mini"
    
    def __init__(self, collection):
        """
        Set up this instance by connecting to OpenAI or DeepSeek, to the Chroma Datastore,
        And setting up the vector encoding model
        """
        self.log("Initializing Frontier Agent")
        gemini = os.getenv("GEMINI_API_KEY")
        if gemini:
            self.client = OpenAI(api_key=gemini, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
            self.MODEL = "models/gemini-2.5-flash-lite-preview-06-17"
            self.log("Frontier Agent is set up with DeepSeek")
        else:
            self.client = OpenAI()
            self.MODEL = "gpt-4o-mini"
            self.log("Frontier Agent is setting up with OpenAI")
        self.collection = collection
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """
        Create context that can be inserted into the prompt
        :param similars: similar products to the one being estimated
        :param prices: prices of the similar products
        :return: text to insert in the prompt that provides context
        """
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message

    def messages_for(self, description: str, similars: List[str], prices: List[float]) -> List[Dict[str, str]]:
        """
        Create the message list to be included in a call to OpenAI
        With the system and user prompt
        :param description: a description of the product
        :param similars: similar products to this one
        :param prices: prices of similar products
        :return: the list of messages in the format expected by OpenAI
        """
        system_message = "You estimate prices of items. Reply only with the price, no explanation"
        user_prompt = self.make_context(similars, prices)
        user_prompt += "And now the question for you:\n\n"
        user_prompt += "How much does this cost?\n\n" + description
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": "Price is $"}
        ]

    def find_similars(self, description: str):
        """
        Return a list of items similar to the given one by looking in the Chroma datastore
        """
        self.log("Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products")
        vector = self.model.encode([description])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
        documents = results['documents'][0][:]
        prices = [m['price'] for m in results['metadatas'][0][:]]
        self.log("Frontier Agent has found similar products")
        return documents, prices

    def get_price(self, s) -> float:
        """
        A utility that plucks a floating point number out of a string
        """
        s = s.replace('$','').replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def price(self, description: str) -> float:
        """
        Make a call to OpenAI or DeepSeek to estimate the price of the described product,
        by looking up 5 similar products and including them in the prompt to give context
        :param description: a description of the product
        :return: an estimate of the price
        """
        documents, prices = self.find_similars(description)
        self.log(f"Frontier Agent is about to call {self.MODEL} with context including 5 similar products")
        time.sleep(4) 
        response = self.client.chat.completions.create(
            model=self.MODEL, 
            messages=self.messages_for(description, documents, prices),
            max_tokens=5
        )
        reply = response.choices[0].message.content
        result = self.get_price(reply)
        self.log(f"Frontier Agent completed - predicting ${result:.2f}")
        return result
        
# imports

# imports

# imports

# import os
# import re
# import json
# import math
# import time
# from typing import List, Dict
# from sentence_transformers import SentenceTransformer
# from datasets import load_dataset
# import chromadb
# import google.generativeai as genai
# from google.api_core.exceptions import ResourceExhausted, InvalidArgument
# from items import Item
# from testing import Tester
# from agents.agent import Agent


# class FrontierAgent(Agent):

#     name = "Frontier Agent"
#     color = Agent.BLUE

#     # Try different models based on availability
#     MODELS = [
#         "gemini-1.5-flash",      # More stable, fewer rate limits
#         "gemini-1.5-pro",        # If flash doesn't work
#         "gemini-2.0-flash-exp",  # Experimental but often has higher limits
#         "gemini-1.0-pro"         # Fallback option
#     ]

#     def __init__(self, collection):
#         """
#         Set up this instance by connecting to Gemini API with proper configuration
#         """
#         self.log("Initializing Frontier Agent")
        
#         # Configure API key
#         gemini_api_key = os.getenv("GEMINI_API_KEY")
#         if not gemini_api_key:
#             raise ValueError("GEMINI_API_KEY environment variable not set")
        
#         genai.configure(api_key=gemini_api_key)
        
#         # Initialize other components
#         self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
#         self.collection = collection
        
#         # Find working model
#         self.working_model = self._find_working_model()
#         self.log(f"Frontier Agent is ready with {self.working_model}")

#     def _find_working_model(self):
#         """Find a model that works with current API limits"""
#         for model_name in self.MODELS:
#             try:
#                 # Test the model with a simple request
#                 test_model = genai.GenerativeModel(
#                     model_name,
#                     generation_config=genai.types.GenerationConfig(
#                         max_output_tokens=10,
#                         temperature=0.1,
#                         candidate_count=1
#                     )
#                 )
                
#                 # Quick test
#                 response = test_model.generate_content("Say 'test'")
#                 if response.text:
#                     self.log(f"Successfully tested model: {model_name}")
#                     return model_name
                    
#             except Exception as e:
#                 self.log(f"Model {model_name} failed test: {e}")
#                 continue
        
#         # If no model works, return the first one and hope for the best
#         self.log("No models passed test, using fallback")
#         return self.MODELS[0]

#     def _create_model_instance(self):
#         """Create a new model instance with optimal settings"""
#         return genai.GenerativeModel(
#             self.working_model,
#             generation_config=genai.types.GenerationConfig(
#                 max_output_tokens=50,    # Limit tokens for pricing responses
#                 temperature=0.1,         # Low temperature for consistent pricing
#                 candidate_count=1,       # Only one response needed
#                 top_p=0.9               # Focus on most likely tokens
#             )
#         )

#     def make_context(self, similars: List[str], prices: List[float]) -> str:
#         """Create context with limited length to avoid token limits"""
#         message = "Similar items for price reference:\n"
        
#         # Limit context to prevent token overflow
#         max_items = 3  # Reduced from 5 to stay within limits
#         for i, (similar, price) in enumerate(zip(similars[:max_items], prices[:max_items])):
#             # Truncate long descriptions
#             truncated_similar = similar[:200] + "..." if len(similar) > 200 else similar
#             message += f"{i+1}. {truncated_similar} - ${price:.2f}\n"
        
#         return message

#     def messages_for(self, description: str, similars: List[str], prices: List[float]) -> str:
#         """Create optimized prompt for Gemini"""
#         context = self.make_context(similars, prices)
        
#         # Truncate description if too long
#         desc = description[:300] + "..." if len(description) > 300 else description
        
#         prompt = f"""Price estimation task:

# {context}

# Estimate price for: {desc}

# Response format: $XX.XX (number only, no explanation)"""
        
#         return prompt

#     def find_similars(self, description: str):
#         """Find similar items from vector database"""
#         self.log("Finding similar products from database")
#         vector = self.model.encode([description])
#         results = self.collection.query(
#             query_embeddings=vector.astype(float).tolist(), 
#             n_results=5
#         )
#         documents = results['documents'][0][:]
#         prices = [m['price'] for m in results['metadatas'][0][:]]
#         self.log(f"Found {len(documents)} similar products")
#         return documents, prices

#     def get_price(self, s) -> float:
#         """Extract price from string"""
#         if not s:
#             return 0.0
#         s = s.replace('$', '').replace(',', '').strip()
#         match = re.search(r"[-+]?\d*\.?\d+", s)
#         return float(match.group()) if match else 0.0

#     def call_gemini_with_fallback(self, prompt: str) -> str:
#         """Call Gemini API with multiple fallback strategies"""
#         strategies = [
#             # Strategy 1: Try with current model
#             {"model": self.working_model, "delay": 0},
#             # Strategy 2: Wait and retry
#             {"model": self.working_model, "delay": 5},
#             # Strategy 3: Try with different model
#             {"model": "gemini-1.5-flash", "delay": 0},
#             # Strategy 4: Try with minimal model
#             {"model": "gemini-1.0-pro", "delay": 0}
#         ]
        
#         for i, strategy in enumerate(strategies):
#             try:
#                 if strategy["delay"] > 0:
#                     self.log(f"Waiting {strategy['delay']} seconds before retry...")
#                     time.sleep(strategy["delay"])
                
#                 model = genai.GenerativeModel(
#                     strategy["model"],
#                     generation_config=genai.types.GenerationConfig(
#                         max_output_tokens=20,
#                         temperature=0.1,
#                         candidate_count=1
#                     )
#                 )
                
#                 self.log(f"Trying API call with {strategy['model']} (attempt {i+1})")
#                 response = model.generate_content(prompt)
                
#                 if response.text:
#                     return response.text.strip()
                    
#             except ResourceExhausted as e:
#                 self.log(f"ResourceExhausted on attempt {i+1}: {e}")
#                 if i == len(strategies) - 1:
#                     raise e
#                 continue
                
#             except Exception as e:
#                 self.log(f"Error on attempt {i+1}: {e}")
#                 if i == len(strategies) - 1:
#                     raise e
#                 continue
        
#         return ""

#     def price(self, description: str) -> float:
#         """Estimate price with comprehensive error handling"""
#         try:
#             # Get similar items
#             documents, prices = self.find_similars(description)
            
#             if not documents or not prices:
#                 self.log("No similar items found, returning default price")
#                 return 25.0  # Default price
            
#             # Create prompt
#             prompt = self.messages_for(description, documents, prices)
            
#             # Try API call with fallbacks
#             try:
#                 reply = self.call_gemini_with_fallback(prompt)
#                 result = self.get_price(reply)
                
#                 if result > 0:
#                     self.log(f"Frontier Agent completed - predicting ${result:.2f}")
#                     return result
#                 else:
#                     raise ValueError("Could not parse price from response")
                    
#             except Exception as api_error:
#                 self.log(f"All API attempts failed: {api_error}")
#                 # Fallback to average of similar items
#                 fallback_price = sum(prices) / len(prices)
#                 self.log(f"Using fallback price (average): ${fallback_price:.2f}")
#                 return fallback_price
                
#         except Exception as e:
#             self.log(f"Complete failure in price estimation: {e}")
#             return 25.0  # Ultimate fallback

#     def price_offline(self, description: str) -> float:
#         """Price estimation without API calls - uses only similar items"""
#         try:
#             documents, prices = self.find_similars(description)
#             if prices:
#                 avg_price = sum(prices) / len(prices)
#                 self.log(f"Offline pricing: ${avg_price:.2f} (average of {len(prices)} similar items)")
#                 return avg_price
#             else:
#                 self.log("No similar items found for offline pricing")
#                 return 25.0
#         except Exception as e:
#             self.log(f"Offline pricing failed: {e}")
#             return 25.0

#     def check_api_status(self):
#         """Check if API is working"""
#         try:
#             model = genai.GenerativeModel("gemini-1.5-flash")
#             response = model.generate_content("Hello")
#             if response.text:
#                 self.log("API is working correctly")
#                 return True
#         except Exception as e:
#             self.log(f"API check failed: {e}")
#             return False