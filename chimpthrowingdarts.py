import datetime
import json
import os
import re
import random
import dotenv
import asyncio
dotenv.load_dotenv()

import numpy as np
import requests

######################### CONSTANTS #########################
# Constants
SUBMIT_PREDICTION = True  # set to True to publish your predictions to Metaculus
USE_EXAMPLE_QUESTIONS = False  # set to True to forecast example questions rather than the tournament questions
NUM_RUNS_PER_QUESTION = 5  # The median forecast is taken between NUM_RUNS_PER_QUESTION runs
SKIP_PREVIOUSLY_FORECASTED_QUESTIONS = True

METACULUS_KEY = os.getenv("METACULUS_KEY")

# The tournament IDs below can be used for testing your bot.
Q4_2024_AI_BENCHMARKING_ID = 32506
Q1_2025_AI_BENCHMARKING_ID = 32627
FALL_2025_AI_BENCHMARKING_ID = "fall-aib-2025"
CURRENT_MINIBENCH_ID = "minibench"

Q4_2024_QUARTERLY_CUP_ID = 3672
Q1_2025_QUARTERLY_CUP_ID = 32630
CURRENT_METACULUS_CUP_ID = "metaculus-cup"

AXC_2025_TOURNAMENT_ID = 32564
AI_2027_TOURNAMENT_ID = "ai-2027"

TOURNAMENT_ID = FALL_2025_AI_BENCHMARKING_ID

# The example questions can be used for testing your bot. (note that question and post id are not always the same)
EXAMPLE_QUESTIONS = [  # (question_id, post_id)
    (578, 578),  # Human Extinction - Binary - https://www.metaculus.com/questions/578/human-extinction-by-2100/
    (14333, 14333),  # Age of Oldest Human - Numeric - https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/
    (22427, 22427),  # Number of New Leading AI Labs - Multiple Choice - https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/
    (38195, 38880), # Number of US Labor Strikes Due to AI in 2029 - Discrete - https://www.metaculus.com/c/diffusion-community/38880/how-many-us-labor-strikes-due-to-ai-in-2029/
]


######################### HELPER FUNCTIONS #########################

# @title Helper functions
AUTH_HEADERS = {"headers": {"Authorization": f"Token {METACULUS_KEY}"}}
API_BASE_URL = "https://www.metaculus.com/api"


def post_question_comment(post_id: int, comment_text: str) -> None:
    """
    Post a comment on the question page as the bot user.
    """

    response = requests.post(
        f"{API_BASE_URL}/comments/create/",
        json={
            "text": comment_text,
            "parent": None,
            "included_forecast": True,
            "is_private": True,
            "on_post": post_id,
        },
        **AUTH_HEADERS,  # type: ignore
    )
    if not response.ok:
        raise RuntimeError(response.text)


def post_question_prediction(question_id: int, forecast_payload: dict) -> None:
    """
    Post a forecast on a question.
    """
    url = f"{API_BASE_URL}/questions/forecast/"
    response = requests.post(
        url,
        json=[
            {
                "question": question_id,
                **forecast_payload,
            },
        ],
        **AUTH_HEADERS,  # type: ignore
    )
    print(f"Prediction Post status code: {response.status_code}")
    if not response.ok:
        raise RuntimeError(response.text)


def create_forecast_payload(
    forecast: float | dict[str, float] | list[float],
    question_type: str,
) -> dict:
    """
    Accepts a forecast and generates the api payload in the correct format.

    If the question is binary, forecast must be a float.
    If the question is multiple choice, forecast must be a dictionary that
      maps question.options labels to floats.
    If the question is numeric, forecast must be a dictionary that maps
      quartiles or percentiles to datetimes, or a 201 value cdf.
    """
    if question_type == "binary":
        return {
            "probability_yes": forecast,
            "probability_yes_per_category": None,
            "continuous_cdf": None,
        }
    if question_type == "multiple_choice":
        return {
            "probability_yes": None,
            "probability_yes_per_category": forecast,
            "continuous_cdf": None,
        }
    # numeric or date
    return {
        "probability_yes": None,
        "probability_yes_per_category": None,
        "continuous_cdf": forecast,
    }


def list_posts_from_tournament(
    tournament_id: int | str = TOURNAMENT_ID, offset: int = 0, count: int = 50
) -> list[dict]:
    """
    List (all details) {count} posts from the {tournament_id}
    """
    url_qparams = {
        "limit": count,
        "offset": offset,
        "order_by": "-hotness",
        "forecast_type": ",".join(
            [
                "binary",
                "multiple_choice",
                "numeric",
                "discrete",
            ]
        ),
        "tournaments": [tournament_id],
        "statuses": "open",
        "include_description": "true",
    }
    url = f"{API_BASE_URL}/posts/"
    response = requests.get(url, **AUTH_HEADERS, params=url_qparams)  # type: ignore
    if not response.ok:
        raise Exception(response.text)
    data = json.loads(response.content)
    return data


def get_open_question_ids_from_tournament() -> list[tuple[int, int]]:
    posts = list_posts_from_tournament()

    post_dict = dict()
    for post in posts["results"]:
        if question := post.get("question"):
            # single question post
            post_dict[post["id"]] = [question]

    open_question_id_post_id = []  # [(question_id, post_id)]
    for post_id, questions in post_dict.items():
        for question in questions:
            if question.get("status") == "open":
                print(
                    f"ID: {question['id']}\nQ: {question['title']}\nCloses: "
                    f"{question['scheduled_close_time']}"
                )
                open_question_id_post_id.append((question["id"], post_id))

    return open_question_id_post_id


def get_post_details(post_id: int) -> dict:
    """
    Get all details about a post from the Metaculus API.
    """
    url = f"{API_BASE_URL}/posts/{post_id}/"
    print(f"Getting details for {url}")
    response = requests.get(
        url,
        **AUTH_HEADERS,  # type: ignore
    )
    if not response.ok:
        raise Exception(response.text)
    details = json.loads(response.content)
    return details
    
def get_random_binary_prediction(question_details, num_runs: int):
    prob = random.random()
    return prob, f"Random chimp prediction: {prob:.2f}"

def get_random_numeric_prediction(question_details, num_runs: int):
    # Return a flat CDF of 201 values between 0 and 1
    cdf = np.linspace(0.001, 0.999, 201).tolist()
    return cdf, "Random chimp numeric forecast (uniform CDF)."

def get_random_multiple_choice_prediction(question_details, num_runs: int):
    options = question_details["options"]
    probs = np.random.dirichlet(np.ones(len(options))).tolist()
    forecast = {opt: p for opt, p in zip(options, probs)}
    return forecast, f"Random chimp MC forecast: {forecast}"

################### FORECASTING ###################
def forecast_is_already_made(post_details: dict) -> bool:
    """
    Check if a forecast has already been made by looking at my_forecasts in the question data.

    question.my_forecasts.latest.forecast_values has the following values for each question type:
    Binary: [probability for no, probability for yes]
    Numeric: [cdf value 1, cdf value 2, ..., cdf value 201]
    Multiple Choice: [probability for option 1, probability for option 2, ...]
    """
    try:
        forecast_values = post_details["question"]["my_forecasts"]["latest"][
            "forecast_values"
        ]
        return forecast_values is not None
    except Exception:
        return False


async def forecast_individual_question(
    question_id: int,
    post_id: int,
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> str:
    post_details = get_post_details(post_id)
    question_details = post_details["question"]
    title = question_details["title"]
    question_type = question_details["type"]

    summary_of_forecast = ""
    summary_of_forecast += f"-----------------------------------------------\nQuestion: {title}\n"
    summary_of_forecast += f"URL: https://www.metaculus.com/questions/{post_id}/\n"

    if question_type == "multiple_choice":
        options = question_details["options"]
        summary_of_forecast += f"options: {options}\n"

    if (
        forecast_is_already_made(post_details)
        and skip_previously_forecasted_questions == True
    ):
        summary_of_forecast += f"Skipped: Forecast already made\n"
        return summary_of_forecast

    if question_type == "binary":
        forecast, comment = get_random_binary_prediction(question_details, num_runs_per_question)
    elif question_type == "numeric":
        forecast, comment = get_random_numeric_prediction(question_details, num_runs_per_question)
    elif question_type == "discrete":
        forecast, comment = get_random_numeric_prediction(question_details, num_runs_per_question)
    elif question_type == "multiple_choice":
        forecast, comment = get_random_multiple_choice_prediction(question_details, num_runs_per_question)
    else:
        raise ValueError(f"Unknown question type: {question_type}")

    print(f"-----------------------------------------------\nPost {post_id} Question {question_id}:\n")
    print(f"Forecast for post {post_id} (question {question_id}):\n{forecast}")
    print(f"Comment for post {post_id} (question {question_id}):\n{comment}")

    if question_type == "numeric" or question_type == "discrete":
        summary_of_forecast += f"Forecast: {str(forecast)[:200]}...\n"
    else:
        summary_of_forecast += f"Forecast: {forecast}\n"

    summary_of_forecast += f"Comment:\n```\n{comment[:200]}...\n```\n\n"

    if submit_prediction == True:
        forecast_payload = create_forecast_payload(forecast, question_type)
        post_question_prediction(question_id, forecast_payload)
        post_question_comment(post_id, comment)
        summary_of_forecast += "Posted: Forecast was posted to Metaculus.\n"

    return summary_of_forecast


async def forecast_questions(
    open_question_id_post_id: list[tuple[int, int]],
    submit_prediction: bool,
    num_runs_per_question: int,
    skip_previously_forecasted_questions: bool,
) -> None:
    forecast_tasks = [
        forecast_individual_question(
            question_id,
            post_id,
            submit_prediction,
            num_runs_per_question,
            skip_previously_forecasted_questions,
        )
        for question_id, post_id in open_question_id_post_id
    ]
    forecast_summaries = await asyncio.gather(*forecast_tasks, return_exceptions=True)
    print("\n", "#" * 100, "\nForecast Summaries\n", "#" * 100)

    errors = []
    for question_id_post_id, forecast_summary in zip(
        open_question_id_post_id, forecast_summaries
    ):
        question_id, post_id = question_id_post_id
        if isinstance(forecast_summary, Exception):
            print(
                f"-----------------------------------------------\nPost {post_id} Question {question_id}:\nError: {forecast_summary.__class__.__name__} {forecast_summary}\nURL: https://www.metaculus.com/questions/{post_id}/\n"
            )
            errors.append(forecast_summary)
        else:
            print(forecast_summary)

    if errors:
        print("-----------------------------------------------\nErrors:\n")
        error_message = f"Errors were encountered: {errors}"
        print(error_message)
        raise RuntimeError(error_message)




######################## FINAL RUN #########################
if __name__ == "__main__":
    if USE_EXAMPLE_QUESTIONS:
        open_question_id_post_id = EXAMPLE_QUESTIONS
    else:
        open_question_id_post_id = get_open_question_ids_from_tournament()

    asyncio.run(
        forecast_questions(
            open_question_id_post_id,
            SUBMIT_PREDICTION,
            NUM_RUNS_PER_QUESTION,
            SKIP_PREVIOUSLY_FORECASTED_QUESTIONS,
        )
    )
