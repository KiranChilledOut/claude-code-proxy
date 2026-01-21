# Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies
```bash
# Using UV (recommended)
uv sync

# Or using pip
pip install -r requirements.txt
```

### Step 2: Configure Your Provider

Choose your LLM provider and configure accordingly:

#### OpenAI
```bash
cp .env.example .env
# Edit .env:
# OPENAI_API_KEY="sk-your-openai-key"
# BIG_MODEL="gpt-4o"
# SMALL_MODEL="gpt-4o-mini"
```

#### Azure OpenAI
```bash
cp .env.example .env
# Edit .env:
# OPENAI_API_KEY="your-azure-key"
# OPENAI_BASE_URL="https://your-resource.openai.azure.com/openai/deployments/your-deployment"
# BIG_MODEL="gpt-4"
# SMALL_MODEL="gpt-35-turbo"
```

#### Local Models (Ollama)
```bash
cp .env.example .env
# Edit .env:
# OPENAI_API_KEY="dummy-key"
# OPENAI_BASE_URL="http://localhost:11434/v1"
# BIG_MODEL="llama3.1:70b"
# SMALL_MODEL="llama3.1:8b"
```

### Step 3: Start and Use

```bash
# Start the proxy server
python start_proxy.py

# In another terminal, use with Claude Code
ANTHROPIC_BASE_URL=http://localhost:8083 claude
```

## 🎯 How It Works

| Your Input                                     | Proxy Action               | Result                                          |
| ---------------------------------------------- | -------------------------- | ----------------------------------------------- |
| Claude Code sends `claude-3-5-sonnet-20241022` | Maps to your `BIG_MODEL`   | Uses `gpt-4o` (or whatever you configured)      |
| Claude Code sends `claude-3-5-haiku-20241022`  | Maps to your `SMALL_MODEL` | Uses `gpt-4o-mini` (or whatever you configured) |

## 📋 What You Need

- Python 3.9+
- API key for your chosen provider
- Claude Code CLI installed
- 2 minutes to configure

## 🔧 Default Settings
- Server runs on `http://localhost:8083`
- Maps haiku → SMALL_MODEL, sonnet/opus → BIG_MODEL
- Supports streaming, function calling, images

## 🧪 Test Your Setup
```bash
# Quick test
python src/test_claude_to_openai.py
```

That's it! Now Claude Code can use any OpenAI-compatible provider! 🎉