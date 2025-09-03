import argparse
import logging
import sys

from bots.fall_2025_template_bot import FallTemplateBot2025
from metaculus_utils import get_questions
from reporting_utils import ResearchReport
from llm_utils import GeneralLlm
from analytics_utils import track_event

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def run_research(question, bot, llms):
    """
    Runs research for a given question and returns the research report.
    """
    qid = getattr(question, "id", None)  # ✅ Fix for missing .question_id

    try:
        track_event("Research Started", {"question_id": qid})
        logger.info(f"Running research for question {qid}...")

        report: ResearchReport = bot.run_research_pipeline(question, llms=llms)

        track_event("Research Completed", {"question_id": qid})
        logger.info(f"Research completed for question {qid}")
        return report

    except Exception as e:
        error_message = f"Research failed for question {qid}: {e}"
        logger.error(error_message, exc_info=True)
        track_event("Research Failed", {"question_id": qid, "error": str(e)})
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run forecasting research pipeline")
    parser.add_argument(
        "--question-ids",
        nargs="+",
        type=int,
        help="Optional list of question IDs to research. If not provided, all questions are used.",
    )
    args = parser.parse_args()

    # ✅ LLM setup (explicit but same structure as before)
    llms = {
        "default": GeneralLlm(
            model="openrouter/openai/gpt-4o",
            temperature=0.3,
            timeout=40,
            allowed_tries=2,
        ),
        "researcher": GeneralLlm(
            model="anthropic/claude-3.5-sonnet",
            temperature=0.4,
            timeout=60,
            allowed_tries=2,
        ),
        "forecaster": GeneralLlm(
            model="openrouter/openai/gpt-4.1",
            temperature=0.3,
            timeout=60,
            allowed_tries=2,
        ),
        "parser": GeneralLlm(
            model="openrouter/openai/gpt-4o-mini",
            temperature=0,
            timeout=30,
            allowed_tries=2,
        ),
    }

    # ✅ Bot initialized the same way
    template_bot = FallTemplateBot2025(
        research_reports_per_question=1,
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=False,
        folder_to_save_reports_to="predictions/",
        skip_previously_forecasted_questions=True,
        llms=llms,
    )

    # ✅ Question fetching logic unchanged
    if args.question_ids:
        questions = get_questions(ids=args.question_ids)
    else:
        questions = get_questions()

    logger.info(f"Loaded {len(questions)} questions to research.")

    # ✅ Run pipeline as before
    for q in questions:
        run_research(q, template_bot, llms)
