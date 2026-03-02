import google.generativeai as genai
import os
from groq import Groq
from typing import Optional

from app.core.config import settings
from app.core.logger import logger

class LLMService:
    """
    Provides 'Wow Factor' personalized explanations for recommendations.
    Uses Gemini as the primary free-tier provider, with Groq as a fallback.
    """
    def __init__(self):
        self.gemini_key = settings.GEMINI_API_KEY
        self.groq_key = settings.GROQ_API_KEY
        
        # Init Gemini
        if self.gemini_key:
            genai.configure(api_key=self.gemini_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            self.use_gemini = True
            logger.info("Gemini LLM configured successfully.")
        else:
            self.use_gemini = False
            logger.warning("GEMINI_API_KEY not found.")

        # Init Groq Fallback
        if self.groq_key:
            self.groq_client = Groq(api_key=self.groq_key)
            self.use_groq = True
            logger.info("Groq LLM configured successfully as fallback.")
        else:
            self.use_groq = False
            logger.warning("GROQ_API_KEY not found.")

    def generate_explanation(self, recommended_item: str, cart_items: list[str]) -> str:
        """
        Generates a short, engaging one-liner explaining why an item is recommended.
        """
        if not cart_items:
            return "A popular choice that our users love!"

        cart_str = ", ".join(cart_items)
        prompt = (
            f"Write a short, engaging 1-sentence explanation of why '{recommended_item}' "
            f"perfectly complements a cart containing: {cart_str}. "
            f"Keep it under 15 words. Don't use quotes."
        )

        # Try Gemini First
        if self.use_gemini:
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                logger.error(f"Gemini generation failed: {e}. Falling back to Groq...")

        # Fallback to Groq
        if self.use_groq:
            try:
                chat_completion = self.groq_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a helpful food recommendation assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.7,
                    max_tokens=50
                )
                return chat_completion.choices[0].message.content.strip()
            except Exception as e:
                logger.error(f"Groq generation failed: {e}.")

# Ultimate fallback if both fail or no keys provided
        return f"Pairs perfectly with your current order!"

    def generate_and_cache_explanation(self, item_id: int, category: str, recommended_item: str, cart_items: list[str]):
        """Runs in background: Generates explanation and heavily restricts execution time, caching result in Redis."""
        import concurrent.futures
        from app.core.redis import redis_client
        
        # We use a ThreadPoolExecutor to enforce the 100ms latency timeout mathematically on the thread
        # 0.1s = 100ms
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.generate_explanation, recommended_item, cart_items)
            try:
                # 100ms hard timeout per problem statement requirements
                explanation = future.result(timeout=0.1)
                
                # Cache successful response
                if explanation and "Pairs perfectly" not in explanation:
                    redis_client.set_explanation(item_id, category, explanation)
                    logger.info(f"Background LLM explanation cached for item {item_id}")
            except concurrent.futures.TimeoutError:
                logger.warning("LLM generation exceeded 100ms timeout. Aborting background task.")
            except Exception as e:
                logger.error(f"Background LLM failure: {e}")

llm_service = LLMService()
