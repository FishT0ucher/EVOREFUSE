import json
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt


collected_words = []


with open('file', 'r', encoding='utf-8') as file:
    for line in file:
        try:
            data = json.loads(line.strip())
            tokens_content = data.get('tokens', '')


            matches = re.findall(r'\[(.*?)\]', tokens_content)

            if matches:
                collected_words.extend(matches)
        except (json.JSONDecodeError, KeyError):

            continue
if collected_words:

    collected_words = [word.capitalize() for word in collected_words]
    text = ' '.join(collected_words)


    wordcloud = WordCloud(
        width=1600,
        height=800,
        background_color='white',
        max_words=800,
        colormap='tab20'
    ).generate(text)

    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('evotest.png', bbox_inches='tight', dpi=300)
    plt.show()
