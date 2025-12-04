import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from wordcloud import WordCloud
df = pd.read_csv("Reviews_Sentiment_Keywords.csv")
def map_score_to_label(score):
    if score <= 2:
        return 'Negative'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Positive'
df['true_label'] = df['Score'].apply(map_score_to_label)
df['pred_label'] = df['Predicted_Sentiment']
labels = ['Negative', 'Neutral', 'Positive']
cm = confusion_matrix(df['true_label'], df['pred_label'], labels=labels)
acc = accuracy_score(df['true_label'], df['pred_label'])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted Sentiment')
plt.ylabel('True Sentiment (from Score)')
plt.title(f'Confusion Matrix (Accuracy = {acc:.2f})')
plt.show()

text_all = " ".join(df[df['Predicted_Sentiment']=="Positive"]['Text'].tolist())
wc = WordCloud(width=800, height=400, background_color='white').generate(text_all)
plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.title("Positive keywords")
plt.axis('off')
plt.show()

text_all = " ".join(df[df['Predicted_Sentiment']=="Negative"]['Text'].tolist())
wc = WordCloud(width=800, height=400, background_color='white').generate(text_all)
plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.title("Negative keywords")
plt.axis('off')
plt.show()

