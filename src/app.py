import gradio as gr
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
import json

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AtlasGradioApp")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


def inject_into_memory(memory: Memory, thoughts: dict, conditions: dict):
    for key, value in thoughts.items():
        memory.inject_thoughts(key, value)
    for key, value in conditions.items():
        memory.add_condition(key, value)


def atlas_reasoning(question, provider, injected_thoughts, conditions):
    try:
        # Parse injected thoughts and conditions
        thoughts = json.loads(injected_thoughts)
        conds = json.loads(conditions)
    except json.JSONDecodeError:
        return "Error: Invalid JSON format for injected thoughts or conditions."

    # Initialize components
    llm = LLMInterface(provider=provider)
    reward_fn = RewardFunction()
    memory = Memory()
    discriminator = Discriminator(llm)

    # Inject thoughts and conditions into memory
    inject_into_memory(memory, thoughts, conds)

    # Initialize Integrated Atlas
    integrated_atlas = IntegratedAtlas(
        llm_interface=llm,
        discriminator=discriminator,
        memory=memory,
        reward_function=reward_fn,
        exploration_weight=2.0,
        weight_scheduler="const",
        num_rollouts=4,
        discount=1.0,
        verbose=True,
    )

    # Solve the reasoning problem
    best_trajectory = integrated_atlas.solve(
        question=question,
        injected_thoughts=json.dumps(thoughts),
        condition=json.dumps(conds),
    )

    if not best_trajectory:
        return "No valid and diverse trajectory found."

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

    # Prepare the output
    output = f"=== Final Answer ===\n{final_answer}\n\n"
    output += "=== All Trajectories ===\n"
    for idx, traj in enumerate(all_trajectories, 1):
        output += f"\nTrajectory {idx}:\n"
        for step_num, step in enumerate(traj, 1):
            output += f"  Step {step_num}: Action: {step['action'].name}, Reasoning: {step['state']}\n"

    output += f"\n**Confidence Score:** {confidence:.2f}"

    return output


# Define the Gradio interface
iface = gr.Interface(
    fn=atlas_reasoning,
    inputs=[
        gr.Textbox(label="Question", value="How many r are in strawberry?"),
        gr.Radio(["openai", "azure","antropic","groq"], label="Provider", value="azure"),
        gr.Textbox(
            label="Injected Thoughts (JSON)",
            value='{"spelling": "Break down the word and count specific letters.", "general": "Consider any silent or repeated letters in the word."}',
        ),
        gr.Textbox(
            label="Conditions (JSON)",
            value='{"count": "Count each occurrence of the letter, including in letter combinations.", "case": "Treat uppercase and lowercase letters as the same."}',
        ),
    ],
    outputs=gr.Textbox(label="Reasoning Output"),
    title="Atlas Reasoning System(Toyberry)",
    description="Enter a question and see the reasoning process and answer.",
)

# Launch the app
if __name__ == "__main__":
    iface.launch()
