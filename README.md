# README.md

# Chatbot Repository

Welcome to the Chatbot Repository! This repository contains the code for two different chatbots:

1. **Botmain.py** - A chatbot that assists with programming doubts.
2. **RAGmain.py** - A Retrieval-Augmented Generation (RAG) based chatbot that answers FAQs for the GIVI platform. The data used to answer queries is present in `guvi-faq.html`.

## Getting Started

### Prerequisites

- Python 3.7+
- Streamlit
- Required Python packages (listed in `requirements.txt`)

### Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/chatbot-repo.git
    cd chatbot-repo
    ```

2. **Install the required packages:**

    ```sh
    pip install -r requirements.txt
    ```

3. **API Keys Configuration:**

    - Open `constant.py`.
    - Replace the placeholder API keys with your actual API keys.

### Running the Chatbots

To run either of the chatbots, use the following commands:

- For the programming doubts chatbot:

    ```sh
    streamlit run Botmain.py
    ```

- For the GIVI FAQs chatbot:

    ```sh
    streamlit run RAGmain.py
    ```

### File Descriptions

- **Botmain.py**: This file contains the implementation of the chatbot designed to assist with programming-related questions. It leverages various APIs and machine learning models to provide accurate answers.

- **RAGmain.py**: This file contains the implementation of the RAG-based chatbot for the GIVI platform. It uses the data present in `guvi-faq.html` to answer frequently asked questions.

- **guvi-faq.html**: This HTML file contains the data used by `RAGmain.py` to answer FAQs.

- **constant.py**: This file holds the API keys and other constant values required by the chatbots. Ensure to update the API keys before running the chatbots.

- **requirements.txt**: This file lists all the Python packages required to run the chatbots.

