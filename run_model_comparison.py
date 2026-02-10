
import json
import os
import random
import sys
import importlib
from typing import Any, Callable, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI


from absl import flags

# Access the private _flagvalues module through the flags module
_flagvalues = flags._flagvalues

# Monkey-patch to allow duplicate flag definitions
_original_setitem = _flagvalues.FlagValues.__setitem__


def _patched_setitem(self, name, flag):
    """Patched __setitem__ that allows duplicate flags."""
    # Check if flag already exists by looking in the internal registry
    if hasattr(self, '_flags') and isinstance(self._flags, dict):
        if name in self._flags:
            # Flag exists - silently update it instead of raising error
            self._flags[name] = flag
            return
    # Flag doesn't exist or _flags is not a dict - use original method
    try:
        _original_setitem(self, name, flag)
    except _flagvalues._exceptions.DuplicateFlagError:
        # If duplicate error occurs, update silently
        if hasattr(self, '_flags') and isinstance(self._flags, dict):
            self._flags[name] = flag


# Apply the patch before any imports
_flagvalues.FlagValues.__setitem__ = _patched_setitem

# Now import all modules
import evaluate_trip_planning
import evaluate_meeting_planning
import evaluate_calendar_scheduling

# ============================================================================
# OPENAI CLIENT INITIALIZATION
# ============================================================================

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
_openai_client = None


def _get_openai_client():
    """Get or create OpenAI client instance."""
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment. "
                "Please create a .env file with OPENAI_API_KEY=your_key"
            )
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# ============================================================================
# CONFIGURATION
# ============================================================================

# Models to benchmark (list of model identifiers)
MODELS = [
    "gpt-4o-mini",
    "gpt-3.5-turbo",
    "gpt-4o",
]

# Number of examples to sample per task
SAMPLES_PER_TASK = 10

# Random seed for reproducibility
RANDOM_SEED = 42

# Dataset paths (read-only)
DATASET_PATHS = {
    "trip_planning": "data/trip_planning.json",
    "meeting_planning": "data/meeting_planning.json",
    "calendar_scheduling": "data/calendar_scheduling.json",
}

# ============================================================================
# LLM INTERFACE (PLACEHOLDER - REPLACE WITH ACTUAL LLM CALLS)
# ============================================================================


def call_llm(prompt: str, model: str) -> str:
    """Call an LLM with a prompt and return the response.

    This is a placeholder function. Replace with actual LLM API calls.

    Args:
        prompt: The prompt to send to the model.
        model: The model identifier (e.g., "gpt-4o-mini").

    Returns:
        The model's response as a string.

    Example implementation:
        if model == "gpt-4o-mini":
            return openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )["choices"][0]["message"]["content"]
        elif model == "mistral-7b":
            # Call Mistral API
            ...
    """
    # Call OpenAI API
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


# ============================================================================
# DATA LOADING AND SAMPLING
# ============================================================================


def load_dataset(task_name: str) -> Dict[str, Any]:
    """Load a dataset from JSON file (read-only).

    Args:
        task_name: One of "trip_planning", "meeting_planning", "calendar_scheduling".

    Returns:
        Dictionary mapping example IDs to example data.
    """
    path = DATASET_PATHS[task_name]
    with open(path, "r") as f:
        return json.load(f)


def sample_examples(
    dataset: Dict[str, Any], n: int, seed: int
) -> List[Tuple[str, Any]]:
    """Randomly sample n examples from a dataset.

    Args:
        dataset: Dictionary of examples.
        n: Number of examples to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of (example_id, example_data) tuples.
    """
    random.seed(seed)
    example_ids = list(dataset.keys())
    sampled_ids = random.sample(example_ids, min(n, len(example_ids)))
    return [(example_id, dataset[example_id]) for example_id in sampled_ids]


# ============================================================================
# TASK-SPECIFIC EVALUATION FUNCTIONS
# ============================================================================


def evaluate_trip_planning_task(
    sampled_examples: List[Tuple[str, Any]], model_responses: Dict[str, str]
) -> float:
    """Evaluate trip planning task using existing evaluation function.

    Args:
        sampled_examples: List of (example_id, example_data) tuples.
        model_responses: Dictionary mapping example_id to model response.

    Returns:
        Accuracy score (0.0 to 1.0).
    """
    cities = []
    durations = []
    responses = []

    for example_id, example_data in sampled_examples:
        cities.append(example_data["cities"])
        durations.append(example_data["durations"])
        responses.append(model_responses[example_id])

    return evaluate_trip_planning.compute_score(cities, durations, responses)


def evaluate_meeting_planning_task(
    sampled_examples: List[Tuple[str, Any]], model_responses: Dict[str, str]
) -> float:
    """Evaluate meeting planning task using existing evaluation function.

    Args:
        sampled_examples: List of (example_id, example_data) tuples.
        model_responses: Dictionary mapping example_id to model response.

    Returns:
        Accuracy score (0.0 to 1.0).
    """
    correct_count = 0
    total_count = 0

    for example_id, example_data in sampled_examples:
        # Extract constraints and metadata
        num_people = example_data["num_people"]
        start_location, initial_time = example_data["constraints"][0]
        constraints = evaluate_meeting_planning.process_constraints(
            example_data["constraints"][1:]
        )
        dist_matrix = example_data["dist_matrix"]

        # Get model response and parse it
        pred_plan_text = model_responses[example_id]
        pred_plan = evaluate_meeting_planning.parse_text_plan(pred_plan_text)

        # Validate model plan
        pred_score = evaluate_meeting_planning.validator_from_text(
            pred_plan, constraints, start_location, initial_time, dist_matrix
        )

        # Get golden plan and validate it
        golden_plan = example_data["golden_plan"]
        golden_score = evaluate_meeting_planning.validator_from_text(
            golden_plan, constraints, start_location, initial_time, dist_matrix
        )

        # Compare scores
        if pred_score == golden_score:
            correct_count += 1
        total_count += 1

    return correct_count / total_count if total_count > 0 else 0.0


def evaluate_calendar_scheduling_task(
    sampled_examples: List[Tuple[str, Any]], model_responses: Dict[str, str]
) -> float:
    """Evaluate calendar scheduling task using existing evaluation function.

    Args:
        sampled_examples: List of (example_id, example_data) tuples.
        model_responses: Dictionary mapping example_id to model response.

    Returns:
        Solve rate (0.0 to 1.0).
    """
    responses = []
    solutions = []

    for example_id, example_data in sampled_examples:
        responses.append(model_responses[example_id])
        solutions.append(example_data["golden_plan"])

    return evaluate_calendar_scheduling.compute_solve_rate(responses, solutions)


# Task evaluation dispatcher
TASK_EVALUATORS = {
    "trip_planning": evaluate_trip_planning_task,
    "meeting_planning": evaluate_meeting_planning_task,
    "calendar_scheduling": evaluate_calendar_scheduling_task,
}

# ============================================================================
# MAIN EXPERIMENT RUNNER
# ============================================================================


def run_experiment_for_model(
    model: str,
    task_name: str,
    sampled_examples: List[Tuple[str, Any]],
) -> float:
    """Run experiment for a single model on a single task.

    Args:
        model: Model identifier.
        task_name: Task name.
        sampled_examples: List of sampled examples.

    Returns:
        Accuracy/solve rate score.
    """
    print(f"\n  Running {model} on {task_name}...")

    # Get LLM responses for all sampled examples
    model_responses = {}
    for example_id, example_data in sampled_examples:
        prompt = example_data["prompt_5shot"]
        response = call_llm(prompt, model)
        model_responses[example_id] = response

    # Evaluate using task-specific evaluator
    evaluator = TASK_EVALUATORS[task_name]
    score = evaluator(sampled_examples, model_responses)

    print(f"  {model} on {task_name}: {score:.4f}")
    return score


def run_full_experiment() -> Dict[str, Dict[str, float]]:
    """Run full experiment comparing all models across all tasks.

    Returns:
        Dictionary mapping model names to task scores.
        Format: {model_name: {task_name: score}}
    """
    print("=" * 70)
    print("NATURAL PLAN MODEL COMPARISON EXPERIMENT")
    print("=" * 70)
    print(f"Models: {', '.join(MODELS)}")
    print(f"Samples per task: {SAMPLES_PER_TASK}")
    print(f"Random seed: {RANDOM_SEED}")
    print("=" * 70)

    # Set random seed
    random.seed(RANDOM_SEED)

    # Load and sample datasets
    print("\nLoading datasets and sampling examples...")
    sampled_datasets = {}
    for task_name in DATASET_PATHS.keys():
        dataset = load_dataset(task_name)
        sampled = sample_examples(dataset, SAMPLES_PER_TASK, RANDOM_SEED)
        sampled_datasets[task_name] = sampled
        print(f"  {task_name}: sampled {len(sampled)} examples")

    # Run experiments
    results = {model: {} for model in MODELS}

    for task_name in DATASET_PATHS.keys():
        print(f"\n{'=' * 70}")
        print(f"TASK: {task_name.upper().replace('_', ' ')}")
        print(f"{'=' * 70}")
        sampled_examples = sampled_datasets[task_name]

        for model in MODELS:
            score = run_experiment_for_model(model, task_name, sampled_examples)
            results[model][task_name] = score

    return results


def print_comparison_table(results: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted comparison table of results.

    Args:
        results: Dictionary mapping model names to task scores.
    """
    print("\n" + "=" * 70)
    print("RESULTS: MODEL COMPARISON")
    print("=" * 70)

    # Table header
    header = f"{'Model':<20} | {'Trip (%)':<12} | {'Meeting (%)':<14} | {'Calendar (%)':<15}"
    print(header)
    print("-" * 70)

    # Table rows
    for model in MODELS:
        trip_score = results[model].get("trip_planning", 0.0) * 100
        meeting_score = results[model].get("meeting_planning", 0.0) * 100
        calendar_score = results[model].get("calendar_scheduling", 0.0) * 100

        row = (
            f"{model:<20} | "
            f"{trip_score:>10.1f} | "
            f"{meeting_score:>12.1f} | "
            f"{calendar_score:>13.1f}"
        )
        print(row)

    print("=" * 70)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main entry point for the experiment runner."""
    results = run_full_experiment()
    print_comparison_table(results)


if __name__ == "__main__":
    main()

