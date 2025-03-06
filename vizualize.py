import json
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from wordcloud import WordCloud
import os
import numpy as np
from matplotlib.ticker import FuncFormatter
import pandas as pd

def load_data(file_path):
    """Load data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_event_distribution(data):
    """Analyze distribution of event types"""
    event_types = []
    
    for doc in data:
        if "event_mentions" in doc and doc["event_mentions"]:
            for event in doc["event_mentions"]:
                if "event_type" in event:
                    event_types.append(event["event_type"])
    
    return Counter(event_types)

def analyze_word_frequency(data):
    """Analyze word frequency in the dataset"""
    all_words = []
    
    for doc in data:
        if "tokens" in doc:
            all_words.extend([token.lower() for token in doc["tokens"]])
    
    return Counter(all_words)

def plot_event_distribution(event_counts):
    """Plot distribution of event types"""
    # Set style
    sns.set(style="whitegrid")
    
    # Set up the figure with a specific size
    plt.figure(figsize=(12, 7))
    
    # Create a DataFrame for better visualization
    df = pd.DataFrame({
        'Loại sự kiện': list(event_counts.keys()),
        'Số lượng': list(event_counts.values())
    })
    
    # Sort by count
    df = df.sort_values('Số lượng', ascending=False)
    
    # Create a bar plot with Seaborn
    ax = sns.barplot(x='Loại sự kiện', y='Số lượng', data=df, palette="viridis")
    
    # Add value labels on top of each bar
    for i, count in enumerate(df['Số lượng']):
        ax.text(i, count + max(df['Số lượng'])*0.01, str(count), ha='center', fontweight='bold', fontsize=12)
    
    # Set labels and title with Vietnamese support
    plt.title('Phân bố các loại sự kiện (Event Types Distribution)', fontsize=18, pad=20)
    plt.xlabel('Loại sự kiện (Event Type)', fontsize=14, labelpad=10)
    plt.ylabel('Số lượng (Count)', fontsize=14, labelpad=10)
    
    # Format y-axis to show whole numbers
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: int(y)))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=15, ha='center', fontsize=12)
    
    # Add a grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Tight layout to make sure everything fits
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("event_distribution.png", dpi=300, bbox_inches="tight")
    print("Event distribution chart saved as 'event_distribution.png'")
    
    # Show the plot
    plt.show()

def plot_top_words(word_counts, top_n=10, exclude_stopwords=True):
    """Plot top N most frequent words"""
    # Vietnamese stopwords (common words to exclude if needed)
    stopwords = ['và', 'của', 'có', 'là', 'trong', 'cho', 'được', 'với', 'các', 'những', 'đã', 'về', 'để', 'không', 'tại']
    
    # Filter out stopwords if requested
    if exclude_stopwords:
        word_counts = Counter({word: count for word, count in word_counts.items() if word not in stopwords})
    
    # Get the top N words
    top_words = word_counts.most_common(top_n)
    words, counts = zip(*top_words)
    
    # Set style
    sns.set(style="whitegrid")
    
    # Set up the figure with a specific size
    plt.figure(figsize=(14, 8))
    
    # Create a DataFrame for better visualization
    df = pd.DataFrame({
        'Từ': words,
        'Tần suất': counts
    })
    
    # Create a horizontal bar plot with Seaborn
    ax = sns.barplot(y='Từ', x='Tần suất', data=df, palette="viridis", orient="h")
    
    # Add value labels on the bars
    for i, count in enumerate(counts):
        ax.text(count + max(counts)*0.01, i, str(count), va='center', fontweight='bold', fontsize=11)
    
    # Set labels and title with Vietnamese support
    plt.title(f'Top {top_n} từ xuất hiện nhiều nhất (Most Frequent Words)', fontsize=18, pad=20)
    plt.ylabel('Từ (Word)', fontsize=14, labelpad=10)
    plt.xlabel('Tần suất xuất hiện (Frequency)', fontsize=14, labelpad=10)
    
    # Format x-axis to show whole numbers
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))
    
    # Add a grid for better readability
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Tight layout to make sure everything fits
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("top_words.png", dpi=300, bbox_inches="tight")
    print("Top words chart saved as 'top_words.png'")
    
    # Show the plot
    plt.show()

def create_word_cloud(word_counts):
    """Create a word cloud visualization"""
    # Vietnamese stopwords to exclude
    stopwords = ['và', 'của', 'có', 'là', 'trong', 'cho', 'được', 'với', 'các', 'những', 'đã', 'về', 'để', 'không', 'tại']
    
    # Convert counter to dictionary for wordcloud and filter stopwords
    word_dict = {word: count for word, count in word_counts.items() if word not in stopwords}
    
    # Configure the word cloud
    wordcloud = WordCloud(
        width=2000,
        height=1200,
        background_color='white',
        max_words=200,
        colormap='viridis',
        contour_width=1,
        contour_color='steelblue',
        collocations=False,  # Don't include bigrams
        regexp=r"\w[\w']+",  # Match words
        min_font_size=10,
        max_font_size=300,
        random_state=42,
        font_path='arial.ttf' if os.path.exists('arial.ttf') else None  # Use arial.ttf if available
    ).generate_from_frequencies(word_dict)
    
    # Display the word cloud
    plt.figure(figsize=(20, 12))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Vietnamese Event Text', fontsize=22, pad=20)
    plt.tight_layout()
    
    # Save the word cloud
    plt.savefig("word_cloud.png", dpi=300, bbox_inches="tight")
    print("Word cloud saved as 'word_cloud.png'")
    
    # Show the word cloud
    plt.show()

def generate_model_comparison_chart(models, metrics):
    """Generate a chart comparing different models on metrics like F1, Precision, Recall"""
    # Data
    model_names = list(models.keys())
    
    # Set up the figure with a specific size
    plt.figure(figsize=(14, 8))
    
    # Set width of bars
    barWidth = 0.25
    
    # Set positions of bars on X axis
    r1 = np.arange(len(model_names))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    
    # Create bars
    plt.bar(r1, [models[model]['precision'] for model in model_names], width=barWidth, label='Precision', color='skyblue')
    plt.bar(r2, [models[model]['recall'] for model in model_names], width=barWidth, label='Recall', color='lightgreen')
    plt.bar(r3, [models[model]['f1'] for model in model_names], width=barWidth, label='F1 Score', color='salmon')
    
    # Add xticks on the middle of the group bars
    plt.xlabel('Models', fontsize=14, labelpad=10)
    plt.ylabel('Score', fontsize=14, labelpad=10)
    plt.title('Model Performance Comparison', fontsize=18, pad=20)
    plt.xticks([r + barWidth for r in range(len(model_names))], model_names)
    
    # Create legend
    plt.legend(fontsize=12)
    
    # Create grid for y-axis
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Set y-axis to start from 0
    plt.ylim(0, 1)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
    print("Model comparison chart saved as 'model_comparison.png'")
    
    # Show the plot
    plt.show()

def generate_all_visualizations(file_path):
    """Generate all visualizations from the data"""
    # Load the data
    try:
        data = load_data(file_path)
        print(f"Loaded data with {len(data)} documents")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Analyze event distribution
    event_counts = analyze_event_distribution(data)
    print(f"Found {sum(event_counts.values())} events of {len(event_counts)} different types")
    
    # Analyze word frequency
    word_counts = analyze_word_frequency(data)
    print(f"Found {len(word_counts)} unique words")
    
    # Create visualizations
    plot_event_distribution(event_counts)
    plot_top_words(word_counts, top_n=10, exclude_stopwords=True)
    create_word_cloud(word_counts)
    

if __name__ == "__main__":
    # Path to the tagged events file
    file_path = "tagged_events.json"
    
    # Generate all visualizations
    generate_all_visualizations(file_path)
    
    # If you want to generate model comparison chart specifically
    print("\nGenerate model comparison chart? (y/n)")
    response = input().strip().lower()
    if response == 'y':
        # Sample model performance data - replace with your actual results
        models = {
            'CRF': {'precision': 0.28, 'recall': 0.19, 'f1': 0.23},
            'BiLSTM-CRF': {'precision': 0.30, 'recall': 0.19, 'f1': 0.24},
            'PhoBERT': {'precision': 0.42, 'recall': 0.38, 'f1': 0.40}
        }
        generate_model_comparison_chart(models, ['precision', 'recall', 'f1'])