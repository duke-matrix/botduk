"""
This is the main entry point for the forecasting bot.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from datetime import datetime
from typing import Literal

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    BinaryPrediction,
    PredictedOptionList,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
    structure_output,
)
from mixpanel import Mixpanel
from newsapi import NewsApiClient
from tavily import TavilyClient

# --- Environment Variables ---
# Load environment variables for API keys and tokens
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
MIXPANEL_TOKEN = os.getenv("MIXPANEL_TOKEN")

# --- Analytics Setup ---
# Initialize Mixpanel for event tracking
MIXPANEL_INSTANCE = Mixpanel(MIXPANEL_TOKEN) if MIXPANEL_TOKEN else None
RUN_ID = str(uuid.uuid4())  # Generate a unique ID for this bot run

# --- Logging Setup ---
logger = logging.getLogger(__name__)


def track_event(event_name: str, properties: dict | None = None) -> None:
    """
    Tracks an event using Mixpanel if it's enabled.
    """
    if MIXPANEL_INSTANCE:
        props = properties or {}
        props["distinct_id"] = RUN_ID
        props["run_id"] = RUN_ID
        MIXPANEL_INSTANCE.track(event_name, props)


class FallTemplateBot2025(ForecastBot):
    """
    A template forecasting bot for the Fall 2025 Metaculus AI Tournament.
    """
    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        """
        Conducts research on a given question using Tavily and NewsAPI.
        """
        async with self._concurrency_limiter:
            track_event("Research Started", {"question_id": question.question_id})
            research_summary = "No research was conducted."
            try:
                tavily_results = self.call_tavily(question.question_text)
                news_results = self.call_newsapi(question.question_text)

                research_prompt = clean_indents(
                    f"""
                    Please synthesize the following research materials into a concise summary for a forecaster.
                    The question being researched is: "{question.question_text}"

                    Tavily Search Results:
                    {tavily_results}

                    Recent News Articles:
                    {news_results}

                    Synthesize the information into a clear, brief summary that will help a forecaster understand the key facts and recent developments related to the question.
                    """
                )

                # Use the default LLM to summarize the research
                summarizer_llm = self.get_llm("default", "llm")
                research_summary = await summarizer_llm.invoke(research_prompt)
                track_event("Research Successful", {"question_id": question.question_id, "summary_length": len(research_summary)})
            except Exception as e:
                error_message = f"Research failed for question {question.question_id}: {e}"
                logger.error(error_message)
                track_event("Research Failed", {"question_id": question.question_id, "error": str(e)})

            logger.info(f"Research for URL {question.page_url}:\n{research_summary}")
            return research_summary

    def call_tavily(self, query: str) -> str:
        """
        Performs a search using the Tavily API.
        """
        if not TAVILY_API_KEY:
            return "Tavily API key not set."
        try:
            tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
            response = tavily_client.search(query=query, search_depth="advanced")
            return "\n".join([f"- {res['content']}" for res in response["results"]])
        except Exception as e:
            return f"Tavily search failed: {e}"

    def call_newsapi(self, query: str) -> str:
        """
        Fetches recent news articles using the NewsAPI.
        """
        if not NEWSAPI_API_KEY:
            return "NewsAPI key not set."
        try:
            newsapi = NewsApiClient(api_key=NEWSAPI_API_KEY)
            articles = newsapi.get_everything(q=query, language="en", sort_by="relevancy", page_size=5)
            return "\n".join([f"- {article['title']}: {article['description']}" for article in articles["articles"]])
        except Exception as e:
            return f"NewsAPI search failed: {e}"

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster. Your task is to predict the outcome of the following binary question.

            Question: {question.question_text}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}
            Research Summary: {research}
            Today's Date: {datetime.now().strftime("%Y-%m-%d")}

            First, provide a step-by-step reasoning for your forecast. Consider the status quo, potential scenarios for "Yes" and "No" outcomes, and the time remaining.
            Finally, state your final probability as a percentage: "Probability: ZZ%"
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction, model=self.get_llm("parser", "llm")
        )
        decimal_pred = max(0.01, min(0.99, binary_prediction.prediction_in_decimal))
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {decimal_pred}"
        )
        return ReasonedPrediction(prediction_value=decimal_pred, reasoning=reasoning)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            You are a professional forecaster. Your task is to predict the outcome of the following multiple-choice question.

            Question: {question.question_text}
            Options: {question.options}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}
            Research Summary: {research}
            Today's Date: {datetime.now().strftime("%Y-%m-%d")}

            First, provide a step-by-step reasoning for your forecast. Consider the status quo and potential unexpected outcomes.
            Finally, assign a probability to each option in the format:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            """
        )
        parsing_instructions = clean_indents(
            f"""
            Ensure all option names match exactly from this list: {question.options}.
            Remove any "Option" prefixes if they are not part of the official names.
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        predicted_option_list: PredictedOptionList = await structure_output(
            text_to_structure=reasoning,
            output_type=PredictedOptionList,
            model=self.get_llm("parser", "llm"),
            additional_instructions=parsing_instructions,
        )
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {predicted_option_list}"
        )
        return ReasonedPrediction(
            prediction_value=predicted_option_list, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            You are a professional forecaster. Your task is to provide a probability distribution for the following numeric question.

            Question: {question.question_text}
            Unit: {question.unit_of_measure or "Not stated"}
            Background: {question.background_info}
            Resolution Criteria: {question.resolution_criteria}
            Fine Print: {question.fine_print}
            {lower_bound_message}
            {upper_bound_message}
            Research Summary: {research}
            Today's Date: {datetime.now().strftime("%Y-%m-%d")}

            First, provide a step-by-step reasoning. Consider the status quo, trends, expert opinions, and high/low scenarios.
            Finally, state your final answer as a series of percentiles:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")
        percentile_list: list[Percentile] = await structure_output(
            reasoning, list[Percentile], model=self.get_llm("parser", "llm")
        )
        prediction = NumericDistribution.from_question(percentile_list, question)
        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction.declared_percentiles}"
        )
        return ReasonedPrediction(prediction_value=prediction, reasoning=reasoning)

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        upper_bound = question.nominal_upper_bound or question.upper_bound
        lower_bound = question.nominal_lower_bound or question.lower_bound

        upper_bound_message = (
            f"The question creator suggests the outcome is likely not higher than {upper_bound}."
            if question.open_upper_bound
            else f"The outcome cannot be higher than {upper_bound}."
        )

        lower_bound_message = (
            f"The question creator suggests the outcome is likely not lower than {lower_bound}."
            if question.open_lower_bound
            else f"The outcome cannot be lower than {lower_bound}."
        )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

    parser = argparse.ArgumentParser(
        description="Run the FallTemplateBot2025 forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "metaculus_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "metaculus_cup", "test_questions"] = args.mode
    
    track_event("Run Started", {"mode": run_mode})

    # --- MODIFIED SECTION ---
    # The bot is now configured to save reports locally instead of publishing them.
    template_bot = FallTemplateBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,  # Set to False to disable publishing
        folder_to_save_reports_to="predictions/",  # Specify the directory to save reports
        skip_previously_forecasted_questions=True,
        llms={
            "default": GeneralLlm(
                model="openrouter/openai/gpt-4o",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "parser": "openrouter/openai/gpt-4o-mini",
        },
    )

    try:
        if run_mode == "tournament":
            seasonal_reports = asyncio.run(
                template_bot.forecast_on_tournament(
                    MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
                )
            )
            minibench_reports = asyncio.run(
                template_bot.forecast_on_tournament(
                    MetaculusApi.CURRENT_MINIBENCH_ID, return_exceptions=True
                )
            )
            forecast_reports = seasonal_reports + minibench_reports
        elif run_mode == "metaculus_cup":
            template_bot.skip_previously_forecasted_questions = False
            forecast_reports = asyncio.run(
                template_bot.forecast_on_tournament(
                    MetaculusApi.CURRENT_METACULUS_CUP_ID, return_exceptions=True
                )
            )
        elif run_mode == "test_questions":
            EXAMPLE_QUESTIONS = [
                "https://www.metaculus.com/questions/578/human-extinction-by-2100/",
                "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",
                "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",
                "https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/",
            ]
            template_bot.skip_previously_forecasted_questions = False
            questions = [
                MetaculusApi.get_question_by_url(url) for url in EXAMPLE_QUESTIONS
            ]
            forecast_reports = asyncio.run(
                template_bot.forecast_questions(questions, return_exceptions=True)
            )
        template_bot.log_report_summary(forecast_reports)
        track_event("Run Finished Successfully")
    except Exception as e:
        error_message = f"Run failed with error: {e}"
        logger.error(error_message)
        track_event("Run Finished With Errors", {"error": error_message})
