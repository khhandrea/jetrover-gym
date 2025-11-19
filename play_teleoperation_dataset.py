import sys
from pathlib import Path
from pickle import load

from jetrover_gym.manipulation_env import JetRoverManipulationEnv


DATASET_DIR = "teleoperation_dataset"


def main():
    task_name = 'jetrover-pickup-cube'
    episode = 1
    data_path = Path(DATASET_DIR) / task_name / f"{task_name}_episode_{episode}.pkl"

    if len(sys.argv) >= 2:
        data_path = Path(sys.argv[1])

    with data_path.open('rb') as f:
        trajectory = load(f)

    env = JetRoverManipulationEnv()

    print(f"Play {data_path}")
    env.reset()
    for step, pair in enumerate(trajectory):
        action = pair['action']
        print(f"Step {step}: action: {action}")
        env.step(action)
        

if __name__ == "__main__":
    main()
