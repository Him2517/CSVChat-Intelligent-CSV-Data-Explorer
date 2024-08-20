# Conversational AI with CSV Data

This project implements a conversational AI system that can answer questions about data from a CSV file. It uses language models, document embedding, and vector stores to create an interactive question-answering experience.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Code Structure](#code-structure)
5. [Detailed Explanation](#detailed-explanation)

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Installation

1. Clone this repository:
  `git clone <repository-url>`
  `cd <repository-directory>`

2. Install the required packages:
   `pip install langchain pandas python-dotenv faiss-cpu sentence-transformers`

3. Create a `.env` file in the project root and add your Groq API key:
   `GROQ_API_KEY=your_groq_api_key_here`

## Usage

1. Place your CSV file in the `data` directory and name it `WHR_2019.csv`.

2. Run the script:
   `python app.py`

3. Once the script loads and processes the data, you can start asking questions. Type your questions at the prompt and press Enter.

4. To exit the program, type "exit", "quit", or "bye".

## Code Structure

The main script performs the following steps:

1. Loads environment variables and sets up the Groq API key
2. Reads the CSV file and creates document objects
3. Splits the text into chunks
4. Creates embeddings for the text chunks
5. Stores the embeddings in a FAISS vector store
6. Initializes the Groq language model
7. Sets up a conversational retrieval chain
8. Starts an interactive loop for question-answering
