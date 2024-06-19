import json

from model.model import ABD_model
from src.plots import first_zipf_law, second_zipf_law, mendelbrot_law

if __name__ == "__main__":
  model = ABD_model()

  train, test = model.load_training_data(links_path = "resources/training_links.json", data_path = "resources/training_data.json", limit=140)
  model.train_model(train, test, iterations=30)

  reviews = json.load(open("resources/training_data.json", encoding="utf-8"))
  first_zipf_law(reviews=reviews)
  second_zipf_law(reviews=reviews)
  mendelbrot_law(reviews=reviews)