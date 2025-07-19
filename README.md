# Shadow Puppetry Digital Human

A lightweight, dual-agent interaction system for 2D shadow puppetry digital humans, powered by LLMs and designed for low-resource environments.

<img width="1168" height="395" alt="LOGO" src="https://github.com/user-attachments/assets/8633971e-b5b4-4eff-a169-3ced06b35a86" />

  ![Static Badge](https://img.shields.io/badge/python-3.10%2B-green)   ![Static Badge](https://img.shields.io/badge/support-Windows%26Linux-purple)    ![Static Badge](https://img.shields.io/badge/license-MIT-orange)


## 📦 Requirements

Before running the system, make sure the following components are ready:

1. **Pull the Gemma 3 (4B) model via [Ollama](https://ollama.com):**

```bash
ollama pull gemma3:4b
```

2. **Download the VOSK Chinese speech recognition model:**

- Go to [VOSK Models](https://alphacephei.com/vosk/models)
- Download the model named `vosk-model-cn-0.22`
- Place the unzipped folder inside the `model-cn/` directory in this project:

```
project_root/
├── model-cn/
│   └── vosk-model-cn-0.22/
```

---

## 🚀 Run the Digital Human

To start the digital human interaction system with LLM functionality, run:

```bash
python main.py
```

---

## 🕹️ How to Interact

Once the interface is running, you can interact in two modes:

### 📝 Text Mode

- Type Chinese text directly into the input box as a prompt.

### 🎙️ Voice Mode

- Click the **Text Mode** button to switch to voice interaction.
- Press the **Space** key to start recording.
- Press **Space** again to stop recording.
- The recognized speech will appear in the top-left corner of the screen.

---
