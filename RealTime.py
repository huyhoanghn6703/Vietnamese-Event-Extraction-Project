import gradio as gr
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Simulated models and their accuracies
MODELS = {
    "CRF": {"precision": 0.28, "recall": 0.19, "f1": 0.23},
    "BiLSTM-CRF": {"precision": 0.30, "recall": 0.19, "f1": 0.24},
    "PhoBERT": {"precision": 0.42, "recall": 0.38, "f1": 0.40}
}

EVENT_TYPES = ["Leader-Activity", "Policy-Announcement", "Emergency-Event"]

def simulate_processing(text_input, model_choice, progress=gr.Progress()):
    """Simulate processing a text input with the selected model"""
    if not text_input or len(text_input) < 5:
        return "Text input too short", None, None
    
    # Simulate model processing with progress bar
    for i in progress.tqdm(range(10)):
        time.sleep(0.2)  # Simulate processing time
    
    # Randomly select an event type with weighted probabilities based on model performance
    weights = [0.5, 0.3, 0.2]  # More likely Leader-Activity, then Policy, then Emergency
    event_type = random.choices(EVENT_TYPES, weights=weights)[0]
    
    # Create confidence scores with some randomness based on the model's precision
    model_precision = MODELS[model_choice]["precision"]
    confidence_primary = model_precision * (0.8 + 0.4 * random.random())  # Base precision with some randomness
    
    # Ensure confidence is between 0 and 1
    confidence_primary = min(max(confidence_primary, 0.95), 0.1)
    
    # Create confidence scores for all classes, with the selected class having highest confidence
    confidences = {}
    remaining = 1.0 - confidence_primary
    
    for et in EVENT_TYPES:
        if et == event_type:
            confidences[et] = confidence_primary
        else:
            # Distribute remaining probability
            confidences[et] = remaining * random.random() * 0.8  # 0.8 to leave some for "None"
            remaining -= confidences[et]
    
    confidences["None"] = remaining  # Assign remaining confidence to "None"
    
    # Create result text
    result_text = f"Detected Event Type: {event_type}\n\nConfidence Scores:\n"
    for et, conf in confidences.items():
        result_text += f"- {et}: {conf:.2%}\n"
    
    # Create a visualization of confidence scores
    plt.figure(figsize=(8, 5))
    bars = plt.bar(confidences.keys(), confidences.values(), color=sns.color_palette("viridis", len(confidences)))
    plt.xlabel('Event Type')
    plt.ylabel('Confidence')
    plt.title(f'Model Confidence Scores ({model_choice})')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{height:.2%}', ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    
    # Save the figure to a temporary file
    plt.savefig('temp_confidence.png')
    plt.close()
    
    return text_input, result_text, 'temp_confidence.png'

def create_interface():
    """Create the Gradio interface"""
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Vietnamese Event Extraction System")
        
        with gr.Row():
            # Left side - Facebook iframe
            with gr.Column(scale=1):
                gr.Markdown("### Facebook Page")
                # Use HTML component to display the Facebook iframe
                facebook_iframe = gr.HTML("""
                    <!-- Replace this with your actual Facebook Page embed code -->
                    <iframe src="https://www.facebook.com/plugins/page.php?href=https%3A%2F%2Fwww.facebook.com%2Ffacebook&tabs=timeline&width=340&height=500&small_header=false&adapt_container_width=true&hide_cover=false&show_facepile=true&appId" 
                    width="100%" height="500" style="border:none;overflow:hidden" 
                    scrolling="no" frameborder="0" allowfullscreen="true" 
                    allow="autoplay; clipboard-write; encrypted-media; picture-in-picture; web-share"></iframe>
                    <p class="text-sm text-gray-500">You can import and customize your own Facebook Page iframe here</p>
                """)
                
                gr.Markdown("""### Instructions
                1. Enter Vietnamese text in the input field
                2. Select a model from the dropdown
                3. Click "Extract Events" to analyze the text
                4. View results and confidence scores
                """)
            
            # Right side - Content and classification
            with gr.Column(scale=1):
                # Input controls
                text_input = gr.Textbox(
                    label="Enter Vietnamese Text", 
                    placeholder="Nhập văn bản tiếng Việt để trích xuất sự kiện...",
                    lines=5
                )
                
                model_choice = gr.Dropdown(
                    choices=list(MODELS.keys()), 
                    label="Select Model", 
                    value="PhoBERT",
                    interactive=True
                )
                
                run_button = gr.Button("Extract Events", variant="primary")
                
                # Classification results section
                gr.Markdown("### Classification Results")
                result_display = gr.Textbox(label="Event Detection", interactive=False, lines=8)
                confidence_chart = gr.Image(label="Confidence Scores")
        
        # Connect the run button to the processing function
        run_button.click(
            fn=simulate_processing,
            inputs=[text_input, model_choice],
            outputs=[text_input, result_display, confidence_chart]
        )
        
        # Model comparison section
        with gr.Row():
            gr.Markdown("### Model Performance Comparison")
        
        with gr.Row():
            # Create model comparison chart
            comparison_chart = gr.Image(label="Model Comparison")
            
            # Generate and display the comparison chart
            def generate_model_comparison():
                models = list(MODELS.keys())
                precision = [MODELS[m]["precision"] for m in models]
                recall = [MODELS[m]["recall"] for m in models]
                f1 = [MODELS[m]["f1"] for m in models]
                
                x = np.arange(len(models))
                width = 0.25
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(x - width, precision, width, label='Precision', color='#3498db')
                ax.bar(x, recall, width, label='Recall', color='#2ecc71')
                ax.bar(x + width, f1, width, label='F1 Score', color='#e74c3c')
                
                ax.set_ylabel('Score')
                ax.set_title('Model Performance Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(models)
                ax.legend()
                
                # Add value labels
                for i, v in enumerate(precision):
                    ax.text(i - width, v + 0.01, f'{v:.2f}', ha='center')
                for i, v in enumerate(recall):
                    ax.text(i, v + 0.01, f'{v:.2f}', ha='center')
                for i, v in enumerate(f1):
                    ax.text(i + width, v + 0.01, f'{v:.2f}', ha='center')
                
                plt.tight_layout()
                plt.savefig('model_comparison.png')
                plt.close()
                
                return 'model_comparison.png'
            
            # Display the comparison chart automatically
            comparison_chart.value = generate_model_comparison()
            
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)  # Set share=True to create a public link