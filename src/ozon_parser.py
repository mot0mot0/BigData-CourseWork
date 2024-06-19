import json
import time

import undetected_chromedriver as uc
from bs4 import BeautifulSoup


def parse_data(resurses_path: str) -> tuple:
  reviews = []
  pos_counter = 0
  neg_counter = 0

  for product_link in json.load(open(resurses_path, encoding="utf-8")):
    driver = uc.Chrome(version_main=126)
    driver.delete_all_cookies()
    driver.get(product_link)
    time.sleep(1)

    for _ in range(300):
      driver.execute_script("window.scrollBy(0, 200);")
      time.sleep(0.05)

    time.sleep(5)

    html_sourse_code = driver.execute_script("return document.body.innerHTML;")
    soup = BeautifulSoup(html_sourse_code, "html.parser")
    driver.close()
    driver.quit()

    parsed_reviews = soup.find_all(class_="r3w_29")

    for parsed_review in parsed_reviews:
      if(parsed_review.find(class_="q7q_29")):
        text = parsed_review.find(class_="q7q_29").get_text()
        score_marks = parsed_review.find(class_="q5q_29")
        score_marks = score_marks.find_all(style="color: rgb(255, 168, 0);")

        score = len(score_marks)

        if score<=3: neg_counter+= 1 
        else: pos_counter +=1

        reviews.append({"review":text, "estimation": score})


  print(f"Было загружено {len(reviews)} отзывов\nПоложительных: {pos_counter}\nОтрицателных: {neg_counter}")
  
  return(reviews)