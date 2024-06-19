from model.model import ABD_model

if __name__ == "__main__":
  model = ABD_model()

  model.test_model(links_path = "resources/test_links.json", data_path = "resources/test_data.json", limit = 40)