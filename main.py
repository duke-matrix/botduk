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

# Local imports (package-relative)
from botduk.forecasting_tools import (
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
from botduk.bots.fall_2025_template_bot import FallTemplateBot2025

from mixpanel import Mixpanel
from newsapi import NewsApiClient
from tavily import TavilyClient

# --- Environment Variables ---
METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEWSAPI_API_KEY = os.getenv("NEWSAPI_API_KEY")
MIXPANEL_TOKEN = os.getenv("MIXPANEL_TOKEN")

# --- Analytics Setup -
