import os
import json
with open('./val_config.json') as f:
    data = json.load(f)

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

c = Config(data)

for i in range(1, 11):
    c.load_path = f"exp_Att/h_256_inner-step_1/meta_model_{i * 100}_inner_1.ckpt"
    c.adaptation_steps = 1
    c.iter = i * 100
    # dump config
    with open('new.json', 'w') as f:
        json.dump(c.__dict__, f, indent=2)
    os.system("python meta_evaluator.py -c ./new.json")