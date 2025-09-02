from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Literal

import dotenv
import typeguard
from forecasting_tools import (
    ApiFilter,
    Benchmarker,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MonetaryCostManager,
    run_benchmark_streamlit_page,
)
from mixpanel import Mixpanel

# Assuming your bot class is in main.py
from main import TemplateForecaster

dotenv.load_dotenv()
logger = logging.getLogger(__name__)


######################### CONSTANTS & API CLIENTS #########################

# --- Run Tracking ---
RUN_ID = str(uuid.uuid4())  # Generate a unique ID for this script run for tracking

# --- Environment Variables & API Keys ---
# These are needed for the bot's internal logic (research) and for analytics
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
MIXPANEL_TOKEN = os.getenv("MIXPANEL_TOKEN")

# --- API Clients ---
mixpanel_client = Mixpanel(MIXPANEL_TOKEN) if MIXPANEL_TOKEN else None

# --- OpenRouter Configuration ---
# Models used for benchmarking
OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o-mini"
OPENROUTER_ALT_MODEL = "anthropic/claude-3.5-sonnet"


######################### HELPER FUNCTIONS #########################


def track_event(event_name: str, properties: dict | None = None):
    """
    Tracks an event using Mixpanel if the client is configured.
    """
    if not mixpanel_client:
        return

    event_properties = {
        "run_id": RUN_ID,
        "run_type": "benchmark",
        "timestamp": datetime.utcnow().isoformat(),
    }
    if properties:
        event_properties.update(properties)

    try:
        mixpanel_client.track(
            distinct_id=RUN_ID, event_name=event_name, properties=event_properties
        )
        logger.info(f"[Mixpanel] Tracked event: {event_name}")
    except Exception as e:
        logger.error(f"[Mixpanel] Error tracking event: {e}")


async def benchmark_forecast_bot(mode: str) -> None:
    """
    Run a benchmark that compares your forecasts against the community prediction
    """
    track_event("Benchmark Function Started", {"mode": mode})
    number_of_questions = (
        30  # Recommend 100+ for meaningful error bars, but 30 is faster/cheaper
    )
    if mode == "display":
        run_benchmark_streamlit_page()
        return
    elif mode == "run":
        questions = MetaculusApi.get_benchmark_questions(number_of_questions)
    elif mode == "custom":
        one_year_from_now = datetime.now() + timedelta(days=365)
        api_filter = ApiFilter(
            allowed_statuses=["open"],
            allowed_types=["binary"],
            num_forecasters_gte=40,
            scheduled_resolve_time_lt=one_year_from_now,
            includes_bots_in_aggregates=False,
            community_prediction_exists=True,
        )
        questions = await MetaculusApi.get_questions_matching_filter(
            api_filter,
            num_questions=number_of_questions,
            randomly_sample=True,
        )
        for question in questions:
            question.background_info = None  # Test ability to find new information
    else:
        track_event("Benchmark Failed", {"reason": f"Invalid mode: {mode}"})
        raise ValueError(f"Invalid mode: {mode}")

    with MonetaryCostManager() as cost_manager:
        bots = [
            TemplateForecaster(
                # Bot 1: More predictions, using a cost-effective model
                predictions_per_research_report=5,
                llms={
                    "default": GeneralLlm(
                        model=f"openrouter/{OPENROUTER_DEFAULT_MODEL}",
                        temperature=0.3,
                    ),
                },
            ),
            TemplateForecaster(
                # Bot 2: Fewer predictions, using a more powerful model
                predictions_per_research_report=1,
                llms={
                    "default": GeneralLlm(
                        model=f"openrouter/{OPENROUTER_ALT_MODEL}",
                        temperature=0.3,
                    ),
                },
            ),
        ]
        bots = typeguard.check_type(bots, list[ForecastBot])
        benchmarks = await Benchmarker(
            questions_to_use=questions,
            forecast_bots=bots,
            file_path_to_save_reports="benchmarks/",
