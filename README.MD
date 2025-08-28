# The Magic Notebook ✨ - Your Adaptive Offline Learning Companion

![Project Status](https://img.shields.io/badge/status-active-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

"The Magic Notebook" is an innovative educational platform that reimagines personalized learning through adaptive AI. Designed to operate both online and, crucially, offline, it provides a dynamic, interactive learning experience that adapts to each student's unique needs, pacing, and learning style. Inspired by the vision of a ubiquitous "Offline Khan Academy," this project aims to bridge the digital divide and ensure high-quality, engaging education is accessible to everyone, everywhere.

## 🚀 Key Features

- **Offline-First Design:** Core learning functionality available without an internet connection, ideal for underserved communities.
- **Adaptive AI Tutoring (Mr. Delight):** An intelligent, personalized tutor that adjusts content, difficulty, and pace based on real-time student performance and engagement.
- **Dynamic UI Generation:** The AI doesn't just provide text; it constructs the entire interactive user interface on the fly, including:
  - Rich Markdown notes for concise lessons.
  - Short answer questions (with optional math keypads).
  - Multiple-choice questions (MCQs) and checkboxes.
  - Sliders for confidence and feedback.
  - Drag-and-drop reordering tasks.
  - Interactive drawing boards for visual explanations.
- **Persona-Driven Learning:** "Mr. Delight," a wise and enchanted notebook, guides learners through a magical and encouraging journey.
- **"Self-Questioning & Answering" Pedagogy:** Encourages critical thinking by leading students through concepts with interactive questions and guided discoveries.
- **Comprehensive Student Status Tracking:** Monitors learning confidence, interest, patience, effort/focus, and identifies weak concept spots to inform adaptive pathways.
- **Multi-Modal Input Support:** Handles text, multiple-choice, and even image inputs (from drawing boards) to enrich interaction.
- **Flexible LLM Backend:** Supports various large language models (LLMs) via an abstract `ModelProvider` interface, allowing for easy switching between cloud APIs (e.g., NVIDIA's endpoints) and local inference (e.g., Ollama).

## 💡 Technologies Used

- **Backend:** Python (Flask)
- **Frontend:** HTML, CSS, JavaScript (client-side rendering)
- **AI Models:**
  - **Inference:** NVIDIA AI Endpoints (default), Ollama (for local deployment)
  - **Fine-tuning:** Unsloth for efficient LoRA (Low-Rank Adaptation) with `gemma-3n-E4B-unsloth-bnb-4bit` (or similar base models).
- **Data Generation:** Python scripts leveraging LLMs to create synthetic multi-turn conversational data for fine-tuning.
- **Environment Management:** Conda

## 🛠️ Getting Started

Follow these steps to set up and run "The Magic Notebook" on your local machine.

### Prerequisites

- **Git:** For cloning the repository.
- **Conda (Anaconda/Miniconda):** For environment management.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/the-magic-notebook.git
cd the-magic-notebook
```
