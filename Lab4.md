# **Claude AI Lab Series: API Integration & Application Development with Google Colab**

## **Overview of Claude API on Google Colab**
Claude AI by Anthropic offers powerful API access for building intelligent applications. Google Colab provides a free cloud-based Python environment perfect for prototyping Claude-powered applications without local setup.

---

## **Lab 4.1: Setting Up Claude API & Making First API Calls**

### **Learning Objectives:**
1. Create and configure Anthropic API account
2. Set up secure API key management in Google Colab
3. Make basic API calls to Claude models
4. Understand different Claude model capabilities
5. Implement error handling for API requests

### **Step-by-Step Guide:**

#### **Step 1: Get Claude API Access**
```bash
1. Visit: https://console.anthropic.com/
2. Sign up for Anthropic account
3. Navigate to API Keys section
4. Create new API key with appropriate permissions
5. Copy and save your API key securely
```

#### **Step 2: Create New Google Colab Notebook**
```python
# Open Google Colab: https://colab.research.google.com/
# Click "New Notebook"
# Rename notebook: "Lab_4_1_Claude_API_Basics.ipynb"

# Import required libraries
!pip install anthropic python-dotenv
```

#### **Step 3: Secure API Key Setup**
```python
# Cell 1: Secure API Key Setup
from google.colab import userdata
import os

# Save API key to Colab Secrets (Recommended)
# 1. Click on the "Key" icon in left sidebar
# 2. Add new secret: ANTHROPIC_API_KEY = "your_key_here"
# 3. Access securely:

API_KEY = userdata.get('ANTHROPIC_API_KEY')
# OR use environment variable
os.environ['ANTHROPIC_API_KEY'] = API_KEY

# Verify key is loaded
if API_KEY:
    print("✅ API Key loaded successfully")
    print(f"Key length: {len(API_KEY)} characters")
else:
    print("❌ API Key not found. Please add to Colab Secrets.")
```

#### **Step 4: Make First API Call**
```python
# Cell 2: First API Call
from anthropic import Anthropic

# Initialize client
client = Anthropic(api_key=API_KEY)

# Test with simple message
response = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=100,
    temperature=0.7,
    system="You are a helpful assistant.",
    messages=[
        {"role": "user", "content": "Hello Claude! Can you introduce yourself?"}
    ]
)

print("Response:", response.content[0].text)
print("\nResponse metadata:")
print(f"Model: {response.model}")
print(f"Stop reason: {response.stop_reason}")
print(f"Usage - Input tokens: {response.usage.input_tokens}")
print(f"Usage - Output tokens: {response.usage.output_tokens}")
```


### **Use Cases:**
1. **API Integration Testing**: Verify Claude API connectivity
2. **Model Comparison**: Choose optimal model for specific tasks
3. **Cost Estimation**: Calculate token usage for budget planning
4. **Error Recovery**: Build resilient API integration
5. **Quick Prototyping**: Test ideas without building full application

