from transformers import pipeline

model= pipeline(task="summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    summary = model(text, max_new_tokens=130, min_length=30, do_sample=False)
    return summary

if __name__ == "__main__":
    text = (
        "The Transformers library by Hugging Face provides state-of-the-art general-purpose architectures for natural language understanding and generation. "
        "It offers a wide range of pre-trained models that can be easily fine-tuned for various NLP tasks such as text classification, named entity recognition, question answering, and text summarization. "
        "With its user-friendly API, developers can quickly integrate these models into their applications, enabling advanced language processing capabilities with minimal effort."
    )
    summary = summarize_text(text)
    print("Summary:", summary[0]['summary_text'])