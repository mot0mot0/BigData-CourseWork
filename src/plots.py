import math

import matplotlib.pyplot as plt
from collections import Counter

def mendelbrot_law(reviews: list):
    counter = {
        k: v
        for k, v in sorted(
            dict(
                Counter([word for item in reviews for word in item["review"].split()])
            ).items(),
            key=lambda x: x[1],
        )
    }
    res = []
    diagram_data = []

    for k, v in counter.items():
        temp = math.fabs(math.log(len(k), (1 * 0.1) / v))
        res.append(temp)
        if temp >= 0.5:
            diagram_data.append({k: temp})

    print(f"Естественность Языка: {round(sum(res) / len(res), 5)} из 1")

    diagram_data = diagram_data[:10]
    _, ax = plt.subplots()
    ax.pie(
        [value for item in diagram_data for _, value in item.items()],
        labels=[key for item in diagram_data for key, _ in item.items()],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.grid()
    plt.show()


def first_zipf_law(reviews: list):
    counter = {
        k: v
        for k, v in sorted(
            dict(
                Counter([word for item in reviews for word in item["review"].split()])
            ).items(),
            key=lambda x: x[1],
        )
    }

    keys_length = {}

    for key in counter.keys():
        keys_length[key] = len(key)

    sorted_key_length = {
        k: v for k, v in sorted(keys_length.items(), key=lambda item: item[1])
    }

    plt.title("Первый закон Ципфа")
    plt.xlabel("Значение слова")
    plt.ylabel("Частота")
    plt.plot(
        list(sorted_key_length.values()),
        [counter[key] for key in list(sorted_key_length.keys())],
    )
    plt.show()


def second_zipf_law(reviews: list):
    counter = dict(
        Counter([word for item in reviews for word in item["review"].split()])
    )

    plt.title("Второй закон Ципфа")
    plt.xlabel("Частота")
    plt.ylabel("Количество слов")
    plt.plot([length for length in range(len(counter))], sorted(list(counter.values()), reverse=True))
    plt.show()