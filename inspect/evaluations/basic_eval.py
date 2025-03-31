from inspect_ai import Task, task
from inspect_ai.dataset import csv_dataset, json_dataset
from inspect_ai.solver import generate
from inspect_ai.scorer import exact, pattern, includes
from dotenv import load_dotenv
load_dotenv()  

my_data = csv_dataset("/Users/khoaluong/Desktop/CWRU/Projects/evals/inspect/data/basic_math.csv")

my_solver = [
    generate()
]

my_scorer = includes()

@task
def basic_math_eval():
    return Task(
        dataset=my_data,
        solver=my_solver,
        scorer=my_scorer
    )

if __name__ == "__main__":
    from inspect_ai import eval
    # run locally in Python
    results = eval(basic_math_eval(), model="openai/gpt-4o-mini")
    print(results)
