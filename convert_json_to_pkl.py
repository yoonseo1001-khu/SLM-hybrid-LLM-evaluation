import json
import pickle

# LLM json 파일 

json_path = "triplets_llm.json"

with open(json_path, "r") as f:
    data = json.load(f)

pickle.dump(data, open("triplets_llm.pkl", "wb"))

print("Saved triplets_llm.pkl")
