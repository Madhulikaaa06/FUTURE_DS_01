import pandas as pd

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_excel("sentimentdataset.xlsx")
print(df)

df.columns = df.columns.str.strip()

df = df.dropna(subset=['Text', 'Sentiment', 'Platform'])

df['Timestamp'] = pd.to_datetime(df['Timestamp'])

from collections import Counter
import re

def extract_hashtags(text):
    if pd.isnull(text):
        return []
    return re.findall(r'#\w+', text)

all_hashtags = df['Hashtags'].dropna().apply(extract_hashtags)
flat_hashtags = [tag for sublist in all_hashtags for tag in sublist]
hashtag_counts = Counter(flat_hashtags)
print("Top 10 Hashtags Overall:", hashtag_counts.most_common(10))

for platform in df['Platform'].unique():
    platform_tags = df[df['Platform'] == platform]['Hashtags'].dropna().apply(extract_hashtags)
    flat_platform_tags = [tag for sublist in platform_tags for tag in sublist]
    platform_counts = Counter(flat_platform_tags)
    print(f"Top hashtags on {platform}:", platform_counts.most_common(5))

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_excel('sentimentdataset.xlsx')


df = df.drop_duplicates()
df = df.dropna(subset=['Text'])

def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


df['Cleaned_Text'] = df['Text'].apply(clean_text)

print(df[['Text', 'Cleaned_Text']].head(736))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df is already loaded and Timestamp is parsed
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Date'] = df['Timestamp'].dt.date

# Aggregate counts of posts by Date and Sentiment
agg_df = df.groupby(['Date', 'Sentiment']).size().reset_index(name='Count')

plt.figure(figsize=(14,7))
sns.lineplot(data=agg_df, x='Date', y='Count', hue='Sentiment')
plt.title('Sentiment Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Posts')
plt.tight_layout()
plt.show()

# 1. Plot sentiment distribution
sns.countplot(x="Sentiment", data=df)
plt.title("Overall Sentiment Distribution")
plt.show()

# 2. Platform-wise sentiment
plt.figure(figsize=(10, 6))
sns.countplot(x="Platform", hue="Sentiment", data=df)
plt.title("Sentiment by Platform")
plt.show()

# 3. Top Hashtags
from collections import Counter

hashtags = sum([tag.strip().split() for tag in df["Hashtags"]], [])
top_tags = Counter(hashtags).most_common(10)
tags, counts = zip(*top_tags)

plt.figure(figsize=(10,5))
sns.barplot(x=list(tags), y=list(counts))
plt.title("Top 10 Hashtags")
plt.xticks(rotation=45)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_excel('sentimentdataset.xlsx')
plt.figure(figsize=(8,5))
sns.countplot(data=df, x='Platform', palette='Set2')
plt.title('Number of Posts by Platform')
plt.xlabel('Platform')
plt.ylabel('Number of Posts')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (use only the first 20 rows for a small sample)
df = pd.read_excel('sentimentdataset.xlsx').head(20)

# Convert Timestamp to datetime and extract date
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Date'] = df['Timestamp'].dt.date

# Aggregate: count posts by date and sentiment
agg_df = df.groupby(['Date', 'Sentiment']).size().reset_index(name='Count')

# Plot
plt.figure(figsize=(10,6))
sns.lineplot(data=agg_df, x='Date', y='Count', hue='Sentiment', marker='o')
plt.title('Sentiment Trend Over Time (Sample Data)')
plt.xlabel('Date')
plt.ylabel('Number of Posts')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset and take a small sample (first 20 rows)
df = pd.read_excel('sentimentdataset.xlsx').head(20)

# Clean column names (optional but recommended)
df.columns = df.columns.str.strip()

# Count the number of posts for each sentiment in the sample
sentiment_counts = df['Sentiment'].value_counts()

# Create a pie chart
plt.figure(figsize=(4,4))
plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct='%1.1f%%',
    startangle=70,
    colors=['#66b3ff','#ff9999','#99ff99']
)
plt.title('Sentiment Distribution (Sample of 20)')
plt.axis('equal')  # Ensures the pie chart is a circle
plt.show()














