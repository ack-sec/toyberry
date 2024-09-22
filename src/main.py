from core.llm_interface import LLMInterface
from core.actions import Action
from core.reward_function import RewardFunction
from core.memory import Memory
from core.discriminator import Discriminator
from atlas.atlas_concrete_node import ReasoningAtlasNode
from atlas.atlas_reasoning_searcher import ReasoningAtlasSearcher
from atlas.atlas_integration import IntegratedAtlas
from core.visualization import Visualizer
from core.calculate_confidence import ConfidenceCalculator
import logging
from dotenv import load_dotenv
import os
import argparse
import json

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Load environment variables from .env file
load_dotenv()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Atlas-based Reasoning System")
    parser.add_argument(
        "--provider",
        choices=["openai", "azure"],
        default="azure",
        help="LLM provider (default: azure)",
    )
    parser.add_argument(
        "--injected_thoughts",
        type=str,
        default='{"spelling": "Break down the word and count specific letters.", "general": "Consider any silent or repeated letters in the word."}',
        help="JSON string of injected thoughts for the reasoning process",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        default='{"count": "Count each occurrence of the letter, including in letter combinations.", "case": "Treat uppercase and lowercase letters as the same."}',
        help="JSON string of conditions for the reasoning process",
    )
    parser.add_argument(
        "--question",
        type=str,
        default="How many r are in strawberry?",
        help="The question to be answered",
    )
    return parser.parse_args()


def inject_into_memory(memory: Memory, thoughts: dict, conditions: dict):
    for key, value in thoughts.items():
        memory.inject_thoughts(key, value)
    for key, value in conditions.items():
        memory.add_condition(key, value)


def main():
    args = parse_arguments()

    # Configure logging
    logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for detailed logs
    logger = logging.getLogger("AtlasMain")

    if args.provider == "openai":
        if "OPENAI_API_KEY" not in os.environ:
            logger.error("OPENAI_API_KEY is not set in the environment variables.")
            return
    elif args.provider == "azure":
        if "AZURE_OPENAI_KEY" not in os.environ:
            logger.error("AZURE_OPENAI_KEY is not set in the environment variables.")
            return

    llm = LLMInterface(provider=args.provider)

    reward_fn = RewardFunction()
    memory = Memory()
    discriminator = Discriminator(llm)

    # Parse injected thoughts and conditions from JSON strings
    try:
        injected_thoughts = json.loads(args.injected_thoughts)
        conditions = json.loads(args.conditions)
    except json.JSONDecodeError:
        logger.error("Invalid JSON format for injected thoughts or conditions.")
        return

    # Inject thoughts and conditions into memory
    inject_into_memory(memory, injected_thoughts, conditions)

    # Define the reasoning problem
    question = args.question

    # Initialize Integrated Atlas
    integrated_atlas = IntegratedAtlas(
        llm_interface=llm,
        discriminator=discriminator,
        memory=memory,
        reward_function=reward_fn,
        exploration_weight=2.0,  # Increased for more exploration
        weight_scheduler="const",
        num_rollouts=4,
        discount=1.0,
        verbose=True,  # Set to True for detailed logs
    )

    # Solve the reasoning problem
    best_trajectory = integrated_atlas.solve(
        question=question,
        injected_thoughts=json.dumps(injected_thoughts),
        condition=json.dumps(conditions),
    )

    if not best_trajectory:
        logger.warning("No valid and diverse trajectory found.")
        return

    # Extract the answer from the final step
    final_answer = next(
        (
            step["state"]
            for step in reversed(best_trajectory)
            if "ANSWER:" in step["state"].upper()
        ),
        best_trajectory[-1]["state"],
    )

    # Calculate confidence
    all_trajectories = memory.get_trajectories(question)
    confidence_calculator = ConfidenceCalculator()
    confidence = confidence_calculator.calculate_confidence(all_trajectories)

    # Display the final answer
    print("\n=== Final Answer ===")
    print(final_answer)

    # Display all trajectories
    print("\n=== All Trajectories ===")
    for idx, traj in enumerate(all_trajectories, 1):
        print(f"\nTrajectory {idx}:")
        for step_num, step in enumerate(traj, 1):
            print(
                f"  Step {step_num}: Action: {step['action'].name}, Reasoning: {step['state']}"
            )

    # Display confidence score
    print(f"\n**Confidence Score:** {confidence:.2f}")

    # Visualize the Atlas tree
    visualizer = Visualizer(integrated_atlas.searcher.root_node)
    visualize_path = "atlas_tree.gv"
    visualizer.visualize(
        path=visualize_path, view=False
    )  # Set view=True to open the file automatically
    print(f"\nAtlas Tree visualization saved to {visualize_path}.pdf")


if __name__ == "__main__":
    main()
