import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

dataset_path = "gs://gresearch/robotics/language_table_sim/0.0.1/"

print("Loading builder from:", dataset_path)
builder = tfds.builder_from_directory(dataset_path)

print("Builder:", builder)
print("Info splits:", builder.info.splits)

ds = builder.as_dataset(split="train")
print("Element spec:")
print(ds.element_spec)

for ep_idx, episode in enumerate(tfds.as_numpy(ds.take(1))):
    print("\nEpisode keys:", episode.keys())

    steps = episode["steps"]
    print("steps type:", type(steps))

    # RLDS episodes often store steps as dict of arrays.
    if isinstance(steps, dict):
        print("Step dict keys:", steps.keys())

        if "observation" in steps:
            print("Observation keys:", steps["observation"].keys())
            for k, v in steps["observation"].items():
                try:
                    print("obs", k, type(v), v.shape, v.dtype)
                except Exception as e:
                    print("obs", k, type(v), e)

        if "action" in steps:
            print("Action:", type(steps["action"]), steps["action"].shape, steps["action"].dtype)
            print("First action:", steps["action"][0])

    else:
        print("steps is not dict; first few:")
        for i, step in enumerate(steps):
            print("step", i, step.keys())
            if i >= 2:
                break
