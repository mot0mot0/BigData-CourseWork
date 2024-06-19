import random
import json

import spacy
from spacy.util import minibatch, compounding
import matplotlib.pyplot as plt 
from spacy.training import Example

from src.ozon_parser import parse_data

class ABD_model():
  def __init__(self):
    pass

  def load_training_data(self,
      links_path: str,
      data_path: str,
      split: float = 0.8,
      limit: int = 0
  ) -> tuple:
    with open(data_path, "r+", encoding="utf-8") as json_file:
      try:
        traning_data = json.load(json_file)
      except json.decoder.JSONDecodeError as err:
        traning_data = parse_data(links_path)
        with open(data_path ,"w", encoding="utf-8") as json_file:
          json.dump(traning_data, json_file, ensure_ascii=False)
        

    reviews=[]

    for review in traning_data:
      text = review["review"]
      spacy_label = {
            "cats": {
              "pos": review["estimation"] > 3,
              "neg": review["estimation"] <= 3}
            }
      reviews.append((text, spacy_label))
            
    random.shuffle(reviews)

    if limit:                                  
      reviews = reviews[:limit]              
      
    split = int(len(reviews) * split)

    return reviews[:split], reviews[split:]


  def evaluate_model(self, tokenizer, textcat, test_data: list) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)

    true_predictions_count = 0

    TP, FP, TN, FN = 1e-8, 0, 0, 0

    for i, review in enumerate(textcat.pipe(reviews)):
      true_label = labels[i]['cats']
      score_pos = review.cats['pos'] 
      if true_label['pos']:
          if score_pos >= 0.5:
              TP += 1
              true_predictions_count +=1
          else:
              FN += 1
      else:
          if score_pos >= 0.5:
              FP += 1
          else:
              TN += 1
              true_predictions_count +=1

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f_score = 2 * precision * recall / (precision + recall)

    return {"precision": precision, "recall": recall, "f-score": f_score, "correct_answers": true_predictions_count}


  def train_model(
          self, 
          training_data: list,
          test_data: list,
          iterations: int = 20
          ) -> None:
      nlp = spacy.load('ru_core_news_sm')

      if "textcat" not in nlp.pipe_names:
          nlp.add_pipe("textcat")

      textcat = nlp.get_pipe("textcat")
      textcat.add_label("pos")
      textcat.add_label("neg")

      training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
      ]

      with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()

        print("Начинаем обучение")
        batch_sizes = compounding(
          4.0, 32.0, 1.001
        )

        correct_answers = 0
        iters = 0
        model_precision = []

        for _ in range(iterations):
          loss = {}

          random.shuffle(training_data)

          batches = minibatch(training_data, size=batch_sizes)

          for batch in batches:
            texts, labels = zip(*batch)

            example = []
            for text, label in zip(texts, labels):
              example.append(Example.from_dict(nlp.make_doc(text), label))

            self.history = nlp.update(               
              example,
              drop=0.2,  
              sgd=optimizer,
              losses=loss
            )

            with textcat.model.use_params(optimizer.averages):
              evaluation_results = self.evaluate_model(
                tokenizer=nlp.tokenizer,
                textcat=textcat,
                test_data=test_data
              )
              #print(f"{loss['textcat']:9.6f}\t{evaluation_results['precision']:.3f}\t{evaluation_results['recall']:.3f}\t{evaluation_results['f-score']:.3f}")
              correct_answers += evaluation_results["correct_answers"]
              iters+=len(test_data)

          iteration_precision = correct_answers / iters * 100
          model_precision.append(iteration_precision)

      plt.plot([i for i in range(1, len(model_precision) + 1)], [precision for precision in model_precision])
      plt.xlabel("Эпоха обучения")
      plt.ylabel("Точность модели")
      plt.show() 

      with nlp.use_params(optimizer.averages):
          nlp.to_disk("model_artifacts")


  def test_model(self,
                 links_path: str,
                 data_path: str, 
                 limit: int = 50):
    with open(data_path, "r+", encoding="utf-8") as json_file:
      try:
        test_data = json.load(json_file)
      except json.decoder.JSONDecodeError as err:
        test_data = parse_data(links_path)
        with open(data_path ,"w", encoding="utf-8") as json_file:
          json.dump(test_data, json_file, ensure_ascii=False)

    random.shuffle(test_data)

    if limit:                                  
      test_data = test_data[:limit]

    print("Тестирование модели:")
    try:
      loaded_model = spacy.load("model_artifacts")
    except:
      print("   ERROR: Не удалось получить данные модели, проверьте расположение директории \"model_artifacts\"")
      return
    
    true_predictions_count = 0

    for text in test_data:
      parsed_text = loaded_model(text["review"])

      if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "Положительный отзыв"
        score = parsed_text.cats["pos"]

        if text["estimation"] > 3:
           true_predictions_count+=1
      else:
        prediction = "Негативный отзыв"
        score = parsed_text.cats["neg"]

        if text["estimation"] <= 3:
           true_predictions_count+=1

      print(f"""Текст обзора: {text["review"]}\n
                Оценка ревьюера: {text["estimation"]}\n
                Предсказание: {prediction}\nScore: {score:.3f}
                """)
      
    correct_answers = true_predictions_count / len(test_data) * 100
      
    print(f"""\nТочность ответов модели: {correct_answers:.2f}%\n\n""") 