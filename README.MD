# The Magic Notebook - Your Adaptive Offline Learning Companion ✨

"The Magic Notebook" is an innovative, AI-powered adaptive learning application designed to provide personalized, engaging, and accessible educational content, even without internet connectivity. Embodying the persona of "Mr. Delight," a wise and enchanted notebook, it crafts a unique learning journey for each student, dynamically adjusting lessons and interactions to foster deep understanding and a "flow state" of learning.

## 🚀 Features

*   **Adaptive Learning Persona:** Guided by "Mr. Delight," the AI maintains an encouraging, patient, and curious tone, using intuitive explanations, stories, and analogies.
*   **Dynamic UI Generation:** The user interface is entirely AI-driven, generating interactive elements like short answer fields (with optional math keypads), multiple-choice questions, checkboxes, sliders, reordering tasks, and even drawing boards.
*   **Offline-First Capability:** Designed to function robustly without continuous internet access, leveraging local Large Language Models (LLMs) via Ollama. NVIDIA's cloud inference is also supported as a high-performance alternative.
*   **Personalized Learning Paths:** Utilizes a detailed `status_dictionary` (tracking learning confidence, interest, patience, effort/focus, and weak concept spots) to adapt lesson pace, difficulty, and focus in real-time.
*   **"Self-Questioning & Answering" Pedagogy:** Encourages critical thinking by posing questions that lead to new concepts, fostering curiosity and deeper understanding.
*   **Multimodal Input Support:** Allows users to interact not just with text, but also through drawings, which the AI processes.
*   **Progress Tracking & Session Management:** Monitors student progress through lessons and provides tools for feedback and status checks.

## 🏁 Getting Started

Follow these steps to set up and run "The Magic Notebook" on your local machine.

### Prerequisites

*   Git
*   [Miniconda or Anaconda](https://docs.conda.io/en/latest/miniconda.html) (recommended for environment management)
*   Python 3.9+

### 1. Clone the Repository

First, clone the project repository to your local machine:

```bash
git clone https://github.com/Stanfording/Offline_Khan_Academy.git # Replace with your repo URL
cd Offline_Khan_Academy
```

### 2. Set Up Conda Environment

Create a new Conda environment and activate it:

```bash
conda create -n magic_notebook python
conda activate magic_notebook
```

### 3. Install Python Dependencies

Install all required Python packages using pip:

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (`.env`)

Create a `.env` file in the root directory of the project based on the provided `example.env` (if present) or the structure below. This file specifies which LLM backend the application will use.

By default, the application is configured to use **NVIDIA's cloud inference** for high performance.

```ini
# .env example

# Choose backend: nvidia | ollama | gemini
MODEL_BACKEND=nvidia

# NVIDIA API Key and Model (Default)
NVIDIA_API_KEY=nvapi-********************************* # Replace with your actual NVIDIA API Key
NVIDIA_MODEL=openai/gpt-oss-20b

# --- OR ---

# Gemini API Key and Model (Alternative)
GOOGLE_API_KEY=AIzaS******************************* # Replace with your actual Google API Key
GEMINI_MODEL=gemini-2.5-flash-lite

# --- OR ---

# Ollama Configuration (Local Backend Alternative)
OLLAMA_ENDPOINT=http://localhost:11434
OLLAMA_MODEL=gemma3n-sft:latest # Or gemma3n:e4b if you haven't fine-tuned
OLLAMA_PERSIST_DIR=.ollama_sessions
```

**Important Notes for `.env`:**

*   **`MODEL_BACKEND`**: Set this to `nvidia`, `ollama`, or `gemini` to choose your preferred inference provider.
*   **`NVIDIA_API_KEY`**: Obtain this from the [NVIDIA AI Playground](https://build.nvidia.com/explore/discover).
*   **`GOOGLE_API_KEY`**: Obtain this from the [Google AI Studio](https://makersuite.google.com/app/apikey).
*   **`OLLAMA_MODEL`**: If you're using Ollama, `gemma3n-sft:latest` refers to a fine-tuned version of Gemma 3n. If you haven't fine-tuned, you might use a base model like `gemma3n:e4b` or `gemma:7b`.

## ⚙️ Ollama Setup Guide (for Local Inference)

If you choose `MODEL_BACKEND=ollama` in your `.env` file, you'll need to set up Ollama locally.

1.  **Download and Install Ollama:**
    *   Visit the official [Ollama website](https://ollama.com/download) and download the installer for your operating system.
    *   Follow the installation instructions. Ollama will run as a local server in the background.

2.  **Pull a Model:**
    *   Open your terminal or command prompt.
    *   Pull the desired model. For this project, a `gemma3n` model is recommended. If you haven't fine-tuned, you can start with a base model:
        ```bash
        ollama pull gemma3n:e4b
        # Or a larger variant if your system allows
        # ollama pull gemma:7b
        ```
    *   If you have a fine-tuned model (e.g., from the `finetune` section), you'll load that instead.

3.  **Verify Ollama Server is Running:**
    *   After installation and pulling a model, Ollama usually starts automatically. You can test it by running:
        ```bash
        ollama run gemma3n:e4b "Tell me a fun fact."
        ```
        If it responds, your Ollama server is operational.

4.  **Configure `.env` for Ollama:**
    *   Ensure your `.env` file looks like this (adjusting `OLLAMA_MODEL` if necessary):
        ```ini
        MODEL_BACKEND=ollama
        OLLAMA_ENDPOINT=http://localhost:11434
        OLLAMA_MODEL=gemma3n-sft:latest # Or the model you pulled
        OLLAMA_PERSIST_DIR=.ollama_sessions
        ```

## ▶️ Running the Application

Once your environment is set up and `.env` is configured, you can start "The Magic Notebook":

```bash
python app.py
```

Open your web browser and navigate to `http://localhost:5000` (or the port indicated in your terminal) to start your magical learning journey!

## 🧪 Finetuning (Optional, for Advanced Users)

This project includes a finetuning pipeline to adapt base LLMs to the specific "Mr. Delight" persona and the application's strict JSON output contract.

*   **`finetune_data_generation.py`**: This script generates high-quality, synthetic multi-turn conversations based on the defined prompt, simulating user interactions and AI responses, complete with `status_dictionary` updates and UI commands. This dataset is crucial for teaching the model the desired behavior.
*   **`train.py`**: This script uses the `unsloth` library for efficient LoRA (Low-Rank Adaptation) finetuning. It trains a base LLM (e.g., `gemma-3n-E4B-unsloth-bnb-4bit`) on the generated dataset, specifically teaching it to act as "Mr. Delight" and output the expected JSON format. This code needs to be customized based on different GPU configurations.

This process allows for creating custom, highly specialized models for the application, enhancing performance and adherence to the adaptive learning logic. Refer to the comments within `finetune_data_generation.py` and `train.py` for more details on running this pipeline.

## 🌟 Project Summary

"The Magic Notebook" is more than just an application; it's a vision for a more equitable and engaging educational future. By seamlessly blending cutting-edge AI, dynamic user experience, and a strong commitment to offline accessibility, we're empowering learners worldwide to embark on a truly personalized and magical discovery journey. This project stands as a testament to how thoughtful technical design can break down barriers and unlock the full potential of adaptive learning for everyone.
```