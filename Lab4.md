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

---

## **Lab 4.2: Building a CLI Chat Application with Claude**

### **Learning Objectives:**
1. Create interactive command-line interface with Claude
2. Implement conversation memory and context management
3. Add streaming responses for real-time interaction
4. Customize system prompts for specific use cases
5. Save and load conversation history

### **Step-by-Step Guide:**

#### **Step 1: Set Up Colab Environment**
```python
# Cell 1: Setup and Dependencies
!pip install anthropic rich prompt_toolkit

import os
from google.colab import userdata
from anthropic import Anthropic
from datetime import datetime
import json
```

#### **Step 2: Create Chat Session Manager**
```python
# Cell 2: Chat Session Class
class ClaudeChatSession:
    def __init__(self, api_key, model="claude-3-sonnet-20240229"):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.conversation_history = []
        self.system_prompt = "You are a helpful AI assistant."
        self.max_history = 10  # Keep last 10 messages
    
    def add_message(self, role, content):
        """Add message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_history.append(message)
        
        # Trim history if too long
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history*2:]
    
    def get_response(self, user_input, stream=False):
        """Get response from Claude"""
        self.add_message("user", user_input)
        
        # Prepare messages for API
        api_messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in self.conversation_history[-self.max_history:]
        ]
        
        if stream:
            return self._stream_response(api_messages)
        else:
            return self._get_complete_response(api_messages)
    
    def _get_complete_response(self, messages):
        """Get complete response (non-streaming)"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            temperature=0.7,
            system=self.system_prompt,
            messages=messages
        )
        
        assistant_response = response.content[0].text
        self.add_message("assistant", assistant_response)
        
        return assistant_response
    
    def _stream_response(self, messages):
        """Stream response token by token"""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=1000,
            temperature=0.7,
            system=self.system_prompt,
            messages=messages
        ) as stream:
            full_response = ""
            for text in stream.text_stream:
                full_response += text
                yield text  # Stream token by token
            
            # Save complete response to history
            self.add_message("assistant", full_response)
    
    def save_conversation(self, filename="conversation.json"):
        """Save conversation to file"""
        with open(filename, 'w') as f:
            json.dump({
                "model": self.model,
                "system_prompt": self.system_prompt,
                "messages": self.conversation_history,
                "saved_at": datetime.now().isoformat()
            }, f, indent=2)
    
    def load_conversation(self, filename="conversation.json"):
        """Load conversation from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
            self.model = data.get("model", self.model)
            self.system_prompt = data.get("system_prompt", self.system_prompt)
            self.conversation_history = data.get("messages", [])
```

#### **Step 3: Build Interactive CLI Interface**
```python
# Cell 3: Interactive CLI
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
import time

class ClaudeCLI:
    def __init__(self, api_key):
        self.console = Console()
        self.session = ClaudeChatSession(api_key)
        self.prompt_session = PromptSession()
        self.style = Style.from_dict({
            'prompt': 'ansigreen bold',
            'input': 'ansicyan',
        })
    
    def print_welcome(self):
        """Print welcome message"""
        self.console.print("[bold blue]" + "="*60)
        self.console.print("🤖 Claude AI CLI Chat Assistant".center(60))
        self.console.print("[bold blue]" + "="*60)
        self.console.print("\n[cyan]Commands:")
        self.console.print("  /system [prompt] - Change system prompt")
        self.console.print("  /model [name]    - Change model")
        self.console.print("  /save [file]     - Save conversation")
        self.console.print("  /load [file]     - Load conversation")
        self.console.print("  /clear           - Clear conversation")
        self.console.print("  /history         - Show conversation history")
        self.console.print("  /stats           - Show usage statistics")
        self.console.print("  /help            - Show this help")
        self.console.print("  /quit or /exit   - Exit program")
        self.console.print("\n[green]Start chatting with Claude!")
    
    def handle_command(self, command):
        """Handle special commands"""
        cmd_parts = command.split()
        cmd = cmd_parts[0].lower()
        
        if cmd == "/system" and len(cmd_parts) > 1:
            new_prompt = " ".join(cmd_parts[1:])
            self.session.system_prompt = new_prompt
            return f"✅ System prompt updated to: {new_prompt[:50]}..."
        
        elif cmd == "/model" and len(cmd_parts) > 1:
            new_model = cmd_parts[1]
            self.session.model = new_model
            return f"✅ Model changed to: {new_model}"
        
        elif cmd == "/save":
            filename = cmd_parts[1] if len(cmd_parts) > 1 else "conversation.json"
            self.session.save_conversation(filename)
            return f"✅ Conversation saved to {filename}"
        
        elif cmd == "/load":
            filename = cmd_parts[1] if len(cmd_parts) > 1 else "conversation.json"
            self.session.load_conversation(filename)
            return f"✅ Conversation loaded from {filename}"
        
        elif cmd == "/clear":
            self.session.conversation_history = []
            return "✅ Conversation cleared"
        
        elif cmd == "/history":
            history_text = "\n".join([
                f"{msg['role']}: {msg['content'][:100]}..."
                for msg in self.session.conversation_history[-5:]
            ])
            return f"📜 Recent history:\n{history_text}"
        
        elif cmd == "/stats":
            user_msgs = len([m for m in self.session.conversation_history if m['role'] == 'user'])
            assistant_msgs = len([m for m in self.session.conversation_history if m['role'] == 'assistant'])
            return f"📊 Stats: {user_msgs} user messages, {assistant_msgs} assistant responses"
        
        elif cmd == "/help":
            return """Available commands:
  /system [prompt] - Change system prompt
  /model [name]    - Change model (haiku, sonnet, opus)
  /save [file]     - Save conversation
  /load [file]     - Load conversation
  /clear           - Clear conversation
  /history         - Show conversation history
  /stats           - Show usage statistics
  /help            - Show this help
  /quit or /exit   - Exit program"""
        
        elif cmd in ["/quit", "/exit"]:
            return "EXIT"
        
        else:
            return None  # Not a command, treat as user message
    
    def run(self):
        """Main CLI loop"""
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = self.prompt_session.prompt(
                    "\n[You] > ",
                    style=self.style
                ).strip()
                
                if not user_input:
                    continue
                
                # Check for commands
                if user_input.startswith('/'):
                    result = self.handle_command(user_input)
                    if result == "EXIT":
                        self.console.print("\n👋 Goodbye!")
                        break
                    elif result:
                        self.console.print(f"\n[dim]{result}")
                        continue
                
                # Get response from Claude
                self.console.print("\n[bold blue][Claude]")
                
                # Show typing indicator
                with self.console.status("[bold green]Thinking...", spinner="dots"):
                    response = self.session.get_response(user_input)
                
                # Print response with markdown formatting
                md = Markdown(response)
                self.console.print(md)
                
            except KeyboardInterrupt:
                self.console.print("\n\n⚠️ Interrupted. Type /quit to exit.")
                continue
            except Exception as e:
                self.console.print(f"\n❌ Error: {str(e)}")
                continue

# Initialize and run CLI
if __name__ == "__main__":
    API_KEY = userdata.get('ANTHROPIC_API_KEY')
    cli = ClaudeCLI(API_KEY)
    cli.run()
```

#### **Step 4: Add Streaming Response Feature**
```python
# Cell 4: Streaming Responses
class ClaudeCLIStreaming(ClaudeCLI):
    def get_streaming_response(self, user_input):
        """Get and display streaming response"""
        self.console.print("\n[bold blue][Claude]")
        
        # Start streaming
        response_generator = self.session.get_response(user_input, stream=True)
        
        # Display streaming response
        full_response = ""
        for chunk in response_generator:
            full_response += chunk
            self.console.print(chunk, end="", style="green")
        
        self.console.print()  # New line after streaming
        
        return full_response
    
    def run(self):
        """Main CLI loop with streaming option"""
        self.print_welcome()
        self.console.print("\n[cyan]Mode: [1] Normal [2] Streaming")
        
        mode = self.prompt_session.prompt("Select mode (1/2): ").strip()
        use_streaming = mode == "2"
        
        while True:
            try:
                user_input = self.prompt_session.prompt(
                    "\n[You] > ",
                    style=self.style
                ).strip()
                
                if not user_input:
                    continue
                
                if user_input.startswith('/'):
                    result = self.handle_command(user_input)
                    if result == "EXIT":
                        self.console.print("\n👋 Goodbye!")
                        break
                    elif result:
                        self.console.print(f"\n[dim]{result}")
                        continue
                
                if use_streaming:
                    self.get_streaming_response(user_input)
                else:
                    self.console.print("\n[bold blue][Claude]")
                    with self.console.status("[bold green]Thinking...", spinner="dots"):
                        response = self.session.get_response(user_input)
                    
                    md = Markdown(response)
                    self.console.print(md)
                    
            except KeyboardInterrupt:
                self.console.print("\n\n⚠️ Interrupted. Type /quit to exit.")
                continue
            except Exception as e:
                self.console.print(f"\n❌ Error: {str(e)}")
                continue
```

#### **Step 5: Test and Export Application**
```python
# Cell 5: Run and Export
# Run the CLI application
if __name__ == "__main__":
    API_KEY = userdata.get('ANTHROPIC_API_KEY')
    if not API_KEY:
        print("❌ Please add ANTHROPIC_API_KEY to Colab Secrets")
    else:
        cli = ClaudeCLIStreaming(API_KEY)
        cli.run()

# Export as Python script for local use
code = '''
# Copy the entire class definitions and run code here
# Then download as .py file
'''

with open('claude_cli_app.py', 'w') as f:
    f.write(code)
    
print("✅ Application code saved to claude_cli_app.py")
print("📥 Download link:")
from google.colab import files
files.download('claude_cli_app.py')
```

### **Use Cases:**
1. **Development Testing**: Quick API testing without GUI
2. **System Administration**: CLI assistant for technical tasks
3. **Content Creation**: Streamlined writing assistant
4. **Learning Tool**: Interactive coding/learning companion
5. **Prototype Testing**: Test conversation flows before app development

---

## **Lab 4.3: Building a Streamlit Web App with Claude API**

### **Learning Objectives:**
1. Create interactive web applications with Streamlit
2. Implement real-time chat interfaces
3. Add file upload and processing capabilities
4. Deploy Streamlit apps from Google Colab
5. Create multi-page applications

### **Step-by-Step Guide:**

#### **Step 1: Set Up Streamlit in Colab**
```python
# Cell 1: Setup with Streamlit
!pip install anthropic streamlit pyngrok
!pip install python-dotenv pandas plotly

import os
from google.colab import userdata
from anthropic import Anthropic
import streamlit as st
import pandas as pd
from datetime import datetime
import json
import tempfile

# Get API key
API_KEY = userdata.get('ANTHROPIC_API_KEY')
client = Anthropic(api_key=API_KEY)
```

#### **Step 2: Create Basic Streamlit App**
```python
%%writefile app.py
"""
Lab 4.3: Claude Streamlit Web Application
"""

import streamlit as st
from anthropic import Anthropic
import os
import json
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Claude AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "model" not in st.session_state:
    st.session_state.model = "claude-3-sonnet-20240229"

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "Anthropic API Key:",
        type="password",
        help="Enter your Anthropic API key"
    )
    
    if api_key:
        st.session_state.api_key = api_key
        try:
            client = Anthropic(api_key=api_key)
            st.success("✅ API Key configured")
        except Exception as e:
            st.error(f"❌ Invalid API Key: {str(e)[:100]}")
    
    # Model selection
    st.subheader("🤖 Model Settings")
    model_options = {
        "Claude 3 Haiku (Fast)": "claude-3-haiku-20240307",
        "Claude 3 Sonnet (Balanced)": "claude-3-sonnet-20240229",
        "Claude 3 Opus (Powerful)": "claude-3-opus-20240229",
    }
    
    selected_model = st.selectbox(
        "Choose Model:",
        list(model_options.keys()),
        index=1
    )
    st.session_state.model = model_options[selected_model]
    
    # Generation parameters
    st.subheader("🎛️ Generation Parameters")
    temperature = st.slider(
        "Temperature:",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls randomness (0 = deterministic, 1 = creative)"
    )
    
    max_tokens = st.slider(
        "Max Tokens:",
        min_value=100,
        max_value=4000,
        value=1000,
        step=100,
        help="Maximum response length"
    )
    
    # System prompt
    st.subheader("💬 System Prompt")
    system_prompt = st.text_area(
        "System Instructions:",
        value="You are a helpful AI assistant.",
        height=100,
        help="Instructions that guide Claude's behavior"
    )
    
    # Clear chat button
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main app interface
st.title("🤖 Claude AI Assistant")
st.caption(f"Using: {selected_model}")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to ask?"):
    if not st.session_state.api_key:
        st.error("Please enter your API key in the sidebar!")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                client = Anthropic(api_key=st.session_state.api_key)
                
                # Prepare messages for API
                api_messages = [
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in st.session_state.messages[-10:]  # Last 10 messages
                ]
                
                # Make API call
                response = client.messages.create(
                    model=st.session_state.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=api_messages
                )
                
                assistant_response = response.content[0].text
                
                # Display response
                st.markdown(assistant_response)
                
                # Add to session state
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"Messages: {len(st.session_state.messages)}")
with col2:
    if st.session_state.messages:
        st.caption(f"Last: {st.session_state.messages[-1]['role']}")
with col3:
    st.caption(f"Model: {st.session_state.model}")
```

#### **Step 3: Add File Upload & Processing Feature**
```python
%%writefile app_with_files.py
"""
Enhanced Streamlit App with File Processing
"""

import streamlit as st
from anthropic import Anthropic
import os
import json
import tempfile
from datetime import datetime
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Claude File Processor",
    page_icon="📁",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "processed_content" not in st.session_state:
    st.session_state.processed_content = {}

# File processing functions
def extract_text_from_file(file_path, file_type):
    """Extract text from different file types"""
    try:
        if file_type == "txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_type == "csv":
            df = pd.read_csv(file_path)
            return df.to_string()
        elif file_type == "json":
            with open(file_path, 'r') as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        else:
            return f"File type {file_type} not supported for text extraction"
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Sidebar with file upload
with st.sidebar:
    st.title("📁 File Processing")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload files for Claude to analyze:",
        type=['txt', 'csv', 'json', 'pdf', 'md'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in [f["name"] for f in st.session_state.uploaded_files]:
                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Extract content
                file_ext = uploaded_file.name.split('.')[-1].lower()
                content = extract_text_from_file(tmp_path, file_ext)
                
                # Store in session state
                st.session_state.uploaded_files.append({
                    "name": uploaded_file.name,
                    "type": file_ext,
                    "path": tmp_path,
                    "size": uploaded_file.size,
                    "content_preview": content[:500] + "..." if len(content) > 500 else content
                })
                
                st.session_state.processed_content[uploaded_file.name] = content
                
                st.success(f"✅ Processed: {uploaded_file.name}")
                
                # Clean up temp file
                os.unlink(tmp_path)
    
    # Display uploaded files
    if st.session_state.uploaded_files:
        st.subheader("📄 Uploaded Files")
        for file_info in st.session_state.uploaded_files:
            with st.expander(f"📎 {file_info['name']}"):
                st.caption(f"Type: {file_info['type']} | Size: {file_info['size']} bytes")
                st.text_area("Content preview:", file_info['content_preview'], height=100, disabled=True)
                
                if st.button(f"Use in chat", key=f"use_{file_info['name']}"):
                    st.session_state.file_context = file_info['name']
                    st.rerun()

# Main app
st.title("📁 Claude File Analyzer")
st.caption("Upload files and ask Claude questions about them")

# File context display
if hasattr(st.session_state, 'file_context'):
    st.info(f"📎 Current context: {st.session_state.file_context}")
    
    # Show file content
    with st.expander("View File Content"):
        content = st.session_state.processed_content.get(st.session_state.file_context, "")
        st.text_area("Full content:", content, height=300)

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input with file context
if prompt := st.chat_input("Ask about your uploaded files..."):
    # Prepare context from selected file
    context = ""
    if hasattr(st.session_state, 'file_context'):
        file_name = st.session_state.file_context
        context = f"\n\nReference file: {file_name}\nContent:\n{st.session_state.processed_content.get(file_name, '')[:2000]}"
    
    full_prompt = f"{prompt}{context}"
    
    # Add to messages
    st.session_state.messages.append({"role": "user", "content": full_prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
        if context:
            st.caption(f"📎 With file: {st.session_state.file_context}")
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                # Get API key from environment
                api_key = os.getenv('ANTHROPIC_API_KEY')
                if not api_key:
                    st.error("Please set ANTHROPIC_API_KEY environment variable")
                    st.stop()
                
                client = Anthropic(api_key=api_key)
                
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=1500,
                    temperature=0.7,
                    messages=[
                        {"role": "user", "content": full_prompt}
                    ]
                )
                
                st.markdown(response.content[0].text)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.content[0].text
                })
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Analysis tools
st.sidebar.divider()
st.sidebar.subheader("🔧 Analysis Tools")

if st.sidebar.button("📊 Summarize All Files", use_container_width=True):
    if st.session_state.processed_content:
        with st.spinner("Generating summary..."):
            all_content = "\n\n".join([
                f"=== {name} ===\n{content[:1000]}"
                for name, content in st.session_state.processed_content.items()
            ])
            
            summary_prompt = f"Please summarize the key points from these files:\n\n{all_content}"
            
            try:
                api_key = os.getenv('ANTHROPIC_API_KEY')
                client = Anthropic(api_key=api_key)
                
                response = client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1000,
                    temperature=0.5,
                    messages=[
                        {"role": "user", "content": summary_prompt}
                    ]
                )
                
                with st.expander("📋 Summary Report", expanded=True):
                    st.markdown(response.content[0].text)
            except Exception as e:
                st.error(f"Summary failed: {str(e)}")
```

#### **Step 4: Deploy from Colab using Ngrok**
```python
# Cell 4: Deploy Streamlit App
from pyngrok import ngrok
import threading
import time
import subprocess

def run_streamlit():
    """Run Streamlit app in background"""
    # Set environment variable for API key
    os.environ['ANTHROPIC_API_KEY'] = userdata.get('ANTHROPIC_API_KEY')
    
    # Run Streamlit
    subprocess.run([
        'streamlit', 'run', 'app_with_files.py',
        '--server.port', '8501',
        '--server.address', '0.0.0.0',
        '--browser.serverAddress', '0.0.0.0'
    ])

# Start Streamlit in a separate thread
thread = threading.Thread(target=run_streamlit)
thread.start()

# Wait for Streamlit to start
time.sleep(5)

# Create ngrok tunnel
public_url = ngrok.connect(8501, bind_tls=True)
print(f"🚀 Streamlit app is running!")
print(f"📱 Public URL: {public_url}")
print(f"🔗 Local URL: http://localhost:8501")

# Keep the tunnel open
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n👋 Shutting down...")
    ngrok.kill()
```

#### **Step 5: Create Multi-Page App**
```python
%%writefile multipage_app/
├── app.py              # Main app
├── pages/
│   ├── 1_Chat.py      # Chat page
│   ├── 2_Files.py     # File processing
│   ├── 3_Analysis.py  # Data analysis
│   └── 4_Settings.py  # Settings
└── utils.py           # Shared utilities
```

**File: multipage_app/app.py**
```python
import streamlit as st

st.set_page_config(
    page_title="Claude AI Suite",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Claude AI Application Suite")
st.markdown("""
### Welcome to the Claude AI Application Suite

Select a page from the sidebar to get started:

- **💬 Chat**: Interactive conversation with Claude
- **📁 Files**: Upload and analyze documents
- **📊 Analysis**: Data analysis and visualization
- **⚙️ Settings**: Configure API and preferences
""")

# Initialize session state
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
```

### **Use Cases:**
1. **Document Analysis**: Upload and query documents
2. **Data Analysis**: Process CSV/JSON files with AI
3. **Content Creation**: Web-based writing assistant
4. **Research Assistant**: Analyze research papers
5. **Customer Support**: Build chatbot interface

---

## **Lab 4.4: Claude API Integration for Data Analysis & Visualization**

### **Learning Objectives:**
1. Use Claude for data analysis and insight generation
2. Create automated data visualization reports
3. Implement data cleaning with AI assistance
4. Build predictive analytics with Claude
5. Generate natural language explanations for data

### **Step-by-Step Guide:**

#### **Step 1: Setup Data Analysis Environment**
```python
# Cell 1: Setup
!pip install anthropic pandas numpy matplotlib seaborn plotly scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from anthropic import Anthropic
from google.colab import userdata
import json
import plotly.express as px
import plotly.graph_objects as go

# Initialize Claude
API_KEY = userdata.get('ANTHROPIC_API_KEY')
client = Anthropic(api_key=API_KEY)
```

#### **Step 2: Data Analysis with Claude**
```python
# Cell 2: Data Analysis Assistant
class DataAnalysisAssistant:
    def __init__(self, client):
        self.client = client
    
    def analyze_dataset(self, df, analysis_type="comprehensive"):
        """Get AI analysis of dataset"""
        
        # Convert dataframe to string for analysis
        data_summary = f"""
        Dataset shape: {df.shape}
        Columns: {list(df.columns)}
        Data types:\n{df.dtypes.to_string()}
        Sample data (first 5 rows):\n{df.head().to_string()}
        Basic statistics:\n{df.describe().to_string() if df.select_dtypes(include=[np.number]).shape[1] > 0 else 'No numerical columns'}
        Missing values:\n{df.isnull().sum().to_string()}
        """
        
        prompt = f"""Please analyze this dataset and provide insights:

        {data_summary}

        Provide a comprehensive analysis including:
        1. Data quality assessment
        2. Key patterns and trends
        3. Anomalies or outliers
        4. Recommendations for further analysis
        5. Business/Research implications

        Format the response with clear sections and bullet points.
        """
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.5,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    
    def generate_visualization_code(self, df, visualization_type):
        """Generate Python code for visualizations"""
        
        prompt = f"""Generate Python code using matplotlib/seaborn/plotly to create {visualization_type} visualizations for this dataset:

        Dataset info:
        - Shape: {df.shape}
        - Columns: {list(df.columns)}
        - Data types: {dict(df.dtypes)}
        - Sample: {df.head().to_string()}

        Requirements:
        1. Create appropriate {visualization_type} visualizations
        2. Include proper labels, titles, and legends
        3. Use best practices for data visualization
        4. Include data preprocessing if needed
        5. Add comments explaining each step

        Return only the Python code.
        """
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1500,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    
    def clean_data_with_ai(self, df, cleaning_tasks):
        """Get AI recommendations for data cleaning"""
        
        data_info = f"""
        Dataset: {df.shape}
        Columns: {list(df.columns)}
        Missing values:\n{df.isnull().sum().to_string()}
        Data types:\n{df.dtypes.to_string()}
        Unique values per column:\n{df.nunique().to_string()}
        """
        
        prompt = f"""Help me clean this dataset. Here's the data:

        {data_info}

        I need to perform these cleaning tasks:
        {cleaning_tasks}

        Please provide:
        1. Specific Python code for each cleaning task
        2. Explanation of why each step is necessary
        3. Expected outcome after cleaning
        4. Validation steps to ensure quality

        Focus on practical, efficient solutions.
        """
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.4,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text

# Example usage
df = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=100),
    'sales': np.random.randn(100).cumsum() + 100,
    'customers': np.random.randint(50, 200, 100),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
    'product': np.random.choice(['A', 'B', 'C', 'D'], 100)
})

assistant = DataAnalysisAssistant(client)

# Get analysis
analysis = assistant.analyze_dataset(df)
print("Dataset Analysis:")
print(analysis[:1000])  # First 1000 chars

# Get visualization code
viz_code = assistant.generate_visualization_code(df, "time series and categorical")
print("\nGenerated Visualization Code:")
print(viz_code[:500])
```

#### **Step 3: Execute AI-Generated Code**
```python
# Cell 3: Safe Code Execution
import ast
import textwrap

class SafeCodeExecutor:
    def __init__(self, allowed_modules=None):
        self.allowed_modules = allowed_modules or [
            'pandas', 'numpy', 'matplotlib', 
            'seaborn', 'plotly', 'datetime'
        ]
        self.local_vars = {'df': df}  # Pass your dataframe
    
    def extract_code_from_response(self, response):
        """Extract Python code from Claude's response"""
        # Look for code blocks
        lines = response.split('\n')
        code_lines = []
        in_code_block = False
        
        for line in lines:
            if line.strip().startswith('```python'):
                in_code_block = True
                continue
            elif line.strip().startswith('```'):
                in_code_block = False
                continue
            
            if in_code_block:
                code_lines.append(line)
        
        code = '\n'.join(code_lines)
        
        # If no code blocks found, assume entire response is code
        if not code.strip():
            code = response
        
        return code
    
    def validate_code(self, code):
        """Validate code syntax and safety"""
        try:
            # Parse to check syntax
            ast.parse(code)
            
            # Check for potentially dangerous imports/operations
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split('.')[0] not in self.allowed_modules:
                            return False, f"Import of {alias.name} not allowed"
                elif isinstance(node, ast.ImportFrom):
                    if node.module.split('.')[0] not in self.allowed_modules:
                        return False, f"Import from {node.module} not allowed"
            
            return True, "Code is valid"
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        except Exception as e:
            return False, f"Validation error: {e}"
    
    def execute_code(self, code, show_output=True):
        """Safely execute the generated code"""
        # Extract clean code
        clean_code = self.extract_code_from_response(code)
        
        # Validate
        is_valid, message = self.validate_code(clean_code)
        if not is_valid:
            print(f"❌ Code validation failed: {message}")
            return None
        
        # Create safe execution environment
        exec_globals = {
            '__builtins__': __builtins__,
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            'px': px,
            'go': go
        }
        
        try:
            # Execute in isolated namespace
            exec(clean_code, exec_globals, self.local_vars)
            
            if show_output:
                print("✅ Code executed successfully")
                
                # Check if any plots were created
                if 'plt' in exec_globals and plt.gcf().get_axes():
                    print("📊 Plot generated - displaying...")
                    plt.tight_layout()
                    plt.show()
            
            return self.local_vars
        except Exception as e:
            print(f"❌ Execution error: {e}")
            return None

# Example: Execute visualization code
executor = SafeCodeExecutor()
result = executor.execute_code(viz_code)

if result and 'cleaned_df' in result:
    print(f"Cleaned dataframe shape: {result['cleaned_df'].shape}")
```

#### **Step 4: Automated Report Generation**
```python
# Cell 4: Automated Reporting
class AutomatedReportGenerator:
    def __init__(self, client):
        self.client = client
    
    def generate_report(self, df, report_type="business"):
        """Generate comprehensive data report"""
        
        # Calculate basic statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        # Prepare data summary
        data_summary = f"""
        REPORT DATASET SUMMARY:
        ======================
        Shape: {df.shape[0]} rows, {df.shape[1]} columns
        Time period: {df['date'].min()} to {df['date'].max() if 'date' in df.columns else 'N/A'}
        
        COLUMNS:
        {df.dtypes.to_string()}
        
        KEY STATISTICS:
        {json.dumps(stats, indent=2)}
        
        SAMPLE DATA:
        {df.head(10).to_string()}
        """
        
        prompt = f"""Generate a {report_type} report based on this dataset:

        {data_summary}

        Create a comprehensive report including:

        EXECUTIVE SUMMARY
        - Key findings in 3 bullet points
        - Overall performance assessment
        
        DETAILED ANALYSIS
        - Trend analysis over time
        - Performance by category/region
        - Correlation between key metrics
        - Anomalies and outliers
        
        VISUALIZATION RECOMMENDATIONS
        - Recommended charts and graphs
        - Key metrics to visualize
        - Dashboard layout suggestions
        
        ACTIONABLE INSIGHTS
        - Top 3 opportunities
        - Areas requiring attention
        - Recommendations for next steps
        
        APPENDIX
        - Data quality notes
        - Analysis limitations
        - Suggestions for additional data
        
        Format professionally with clear headings and bullet points.
        """
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=3000,
            temperature=0.4,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    
    def create_presentation(self, df, audience="executive"):
        """Create presentation slides from data"""
        
        prompt = f"""Create a presentation based on this data analysis:

        Dataset: {df.shape}
        Key columns: {list(df.columns)[:10]}
        
        Create a 5-slide presentation for {audience} audience:

        Slide 1: Title Slide
        - Engaging title
        - Key takeaway
        - Presenter info
        
        Slide 2: Overview & Objectives
        - Analysis goals
        - Data sources
        - Methodology
        
        Slide 3: Key Findings
        - Top 3 insights
        - Supporting data points
        - Visual suggestions
        
        Slide 4: Trends & Patterns
        - Time series trends
        - Category comparisons
        - Correlation highlights
        
        Slide 5: Recommendations & Next Steps
        - Actionable recommendations
        - Implementation timeline
        - Success metrics
        
        For each slide provide:
        1. Slide title
        2. 3-5 bullet points
        3. Suggested visualizations
        4. Speaker notes
        """
        
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2500,
            temperature=0.5,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text

# Generate reports
report_generator = AutomatedReportGenerator(client)

business_report = report_generator.generate_report(df, "business")
executive_presentation = report_generator.create_presentation(df, "executive")

print("📊 Business Report (first 1000 chars):")
print(business_report[:1000])
print("\n" + "="*50 + "\n")
print("📈 Executive Presentation:")
print(executive_presentation[:1000])
```

#### **Step 5: Predictive Analytics with Claude**
```python
# Cell 5: Predictive Analysis
class PredictiveAnalyst:
    def __init__(self, client):
        self.client = client
    
    def generate_forecast(self, df, target_column, forecast_periods=30):
        """Generate forecasting code and analysis"""
        
        prompt = f"""Create a forecasting analysis for this time series data:

        Data Info:
        - Target column: {target_column}
        - Date range: {df['date'].min()} to {df['date'].max() if 'date' in df.columns else 'N/A'}
        - Data frequency: Daily
        - Forecast periods: {forecast_periods} days
        
        Dataset sample:
        {df[[target_column, 'date'] if 'date' in df.columns else target_column].head().to_string()}

        Please provide:
        
        1. DATA PREPARATION
        - Code to prepare time series data
        - Handling missing values
        - Feature engineering suggestions
        
        2. MODEL SELECTION
        - Recommended forecasting models (ARIMA, Prophet, etc.)
        - Model selection rationale
        - Hyperparameter tuning approach
        
        3. IMPLEMENTATION CODE
        - Complete Python code for forecasting
        - Model training and validation
        - Forecast generation
        
        4. EVALUATION METRICS
        - Error metrics (MAE, RMSE, MAPE)
        - Backtesting strategy
        - Confidence intervals
        
        5. VISUALIZATION
        - Code to plot historical vs forecast
        - Uncertainty visualization
        - Interactive dashboard suggestions
        
        Focus on production-ready, robust code.
        """
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2500,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    
    def analyze_correlations(self, df):
        """Analyze correlations and relationships"""
        
        # Calculate correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr()
            corr_text = corr_matrix.to_string()
        else:
            corr_text = "Not enough numeric columns for correlation analysis"
        
        prompt = f"""Analyze correlations and relationships in this dataset:

        Dataset: {df.shape}
        Columns: {list(df.columns)}
        
        Correlation Matrix:
        {corr_text}
        
        Please provide:
        
        1. CORRELATION INSIGHTS
        - Strongest positive/negative correlations
        - Unexpected relationships
        - Multicollinearity assessment
        
        2. CAUSAL ANALYSIS
        - Potential cause-effect relationships
        - Confounding variables to consider
        - Granger causality suggestions
        
        3. FEATURE IMPORTANCE
        - Which features are most predictive
        - Feature selection recommendations
        - Dimensionality reduction suggestions
        
        4. BUSINESS/RESEARCH IMPLICATIONS
        - What correlations mean for decision making
        - Actions suggested by the data
        - Further investigation areas
        
        Provide specific, actionable insights.
        """
        
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=2000,
            temperature=0.4,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text

# Run predictive analysis
analyst = PredictiveAnalyst(client)

# Generate forecasting code
if 'sales' in df.columns and 'date' in df.columns:
    forecast_code = analyst.generate_forecast(df, 'sales', 30)
    print("📈 Forecasting Analysis:")
    print(forecast_code[:1000])

# Analyze correlations
correlation_analysis = analyst.analyze_correlations(df)
print("\n🔗 Correlation Analysis:")
print(correlation_analysis[:1000])
```

### **Use Cases:**
1. **Business Intelligence**: Automated insight generation
2. **Research Analysis**: Statistical analysis with natural language
3. **Financial Forecasting**: Time series prediction
4. **Marketing Analytics**: Customer behavior analysis
5. **Scientific Research**: Data interpretation and hypothesis generation

---

## **Lab 4.5: Building Production-Ready Claude API Applications**

### **Learning Objectives:**
1. Implement API rate limiting and error handling
2. Create async API calls for improved performance
3. Build caching mechanisms to reduce API costs
4. Implement monitoring and logging
5. Create reusable API client patterns

### **Step-by-Step Guide:**

#### **Step 1: Production API Client with Retry Logic**
```python
# Cell 1: Production-Ready API Client
import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass
from ratelimit import limits, sleep_and_retry
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """API configuration"""
    api_key: str
    base_url: str = "https://api.anthropic.com/v1"
    max_retries: int = 3
    timeout: int = 30
    rate_limit_per_minute: int = 50  # Anthropic's rate limit

class ProductionClaudeClient:
    """Production-ready Claude API client with retry logic and rate limiting"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = None
        self.request_count = 0
        self.request_timestamps = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = time.time()
        # Remove timestamps older than 1 minute
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if now - ts < 60
        ]
        
        if len(self.request_timestamps) >= self.config.rate_limit_per_minute:
            sleep_time = 60 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        self.request_timestamps.append(now)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda retry_state: None
    )
    async def make_request(self, endpoint: str, data: Dict) -> Optional[Dict]:
        """Make API request with retry logic"""
        self._check_rate_limit()
        
        url = f"{self.config.base_url}/{endpoint}"
        
        try:
            async with self.session.post(url, json=data) as response:
                self.request_count += 1
                
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:  # Rate limited
                    retry_after = int(response.headers.get('retry-after', 5))
                    logger.warning(f"Rate limited. Retrying after {retry_after} seconds")
                    await asyncio.sleep(retry_after)
                    raise Exception("Rate limited")
                elif response.status >= 500:  # Server error
                    logger.error(f"Server error {response.status}")
                    raise Exception(f"Server error: {response.status}")
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error("Request timeout")
            raise
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    async def create_message(self, 
                           model: str = "claude-3-sonnet-20240229",
                           messages: List[Dict] = None,
                           max_tokens: int = 1000,
                           temperature: float = 0.7,
                           system: str = None) -> Optional[Dict]:
        """Create a message with Claude"""
        data = {
            "model": model,
            "messages": messages or [],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if system:
            data["system"] = system
        
        return await self.make_request("messages", data)
    
    async def batch_process(self, 
                          requests: List[Dict],
                          max_concurrent: int = 5) -> List[Optional[Dict]]:
        """Process multiple requests concurrently"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(request):
            async with semaphore:
                return await self.create_message(**request)
        
        tasks = [process_with_semaphore(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)

# Example usage
async def example_usage():
    config = APIConfig(api_key=userdata.get('ANTHROPIC_API_KEY'))
    
    async with ProductionClaudeClient(config) as client:
        # Single request
        response = await client.create_message(
            model="claude-3-haiku-20240307",
            messages=[
                {"role": "user", "content": "Hello, Claude!"}
            ]
        )
        
        if response:
            print(f"Response: {response.get('content', [{}])[0].get('text', '')}")
        
        # Batch processing
        requests = [
            {
                "model": "claude-3-haiku-20240307",
                "messages": [{"role": "user", "content": f"Test message {i}"}],
                "max_tokens": 100
            }
            for i in range(5)
        ]
        
        batch_results = await client.batch_process(requests)
        print(f"\nBatch processed {len(batch_results)} requests")

# Run in Colab
await example_usage()
```

#### **Step 2: Response Caching System**
```python
# Cell 2: Caching System
import hashlib
import pickle
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path

class ResponseCache:
    """Cache API responses to reduce costs and improve performance"""
    
    def __init__(self, cache_dir: str = "./cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        
        # Initialize SQLite cache
        self.db_path = self.cache_dir / "responses.db"
        self.init_database()
    
    def init_database(self):
        """Initialize cache database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    response BLOB,
                    model TEXT,
                    tokens INTEGER,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)")
    
    def generate_key(self, model: str, messages: List[Dict], **kwargs) -> str:
        """Generate cache key from request parameters"""
        key_data = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached response if exists and not expired"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT response, expires_at FROM cache 
                WHERE key = ? AND expires_at > ?
            """, (key, datetime.now()))
            
            row = cursor.fetchone()
            if row:
                response_blob, expires_at = row
                response = pickle.loads(response_blob)
                logger.info(f"Cache hit for key: {key[:16]}... (expires: {expires_at})")
                return response
            
        return None
    
    def set(self, key: str, response: Dict, model: str, tokens: int):
        """Cache API response"""
        expires_at = datetime.now() + self.ttl
        response_blob = pickle.dumps(response)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache 
                (key, response, model, tokens, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (key, response_blob, model, tokens, datetime.now(), expires_at))
        
        logger.info(f"Cached response for key: {key[:16]}... (expires: {expires_at})")
    
    def cleanup(self):
        """Remove expired cache entries"""
        with sqlite3.connect(self.db_path) as conn:
            deleted = conn.execute(
                "DELETE FROM cache WHERE expires_at <= ?",
                (datetime.now(),)
            ).rowcount
            logger.info(f"Cleaned up {deleted} expired cache entries")
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(tokens) as total_tokens,
                    COUNT(CASE WHEN expires_at > ? THEN 1 END) as active,
                    COUNT(CASE WHEN expires_at <= ? THEN 1 END) as expired
                FROM cache
            """, (datetime.now(), datetime.now()))
            
            row = cursor.fetchone()
            return {
                "total_entries": row[0],
                "total_tokens": row[1] or 0,
                "active_entries": row[2],
                "expired_entries": row[3]
            }

class CachedClaudeClient(ProductionClaudeClient):
    """Claude client with response caching"""
    
    def __init__(self, config: APIConfig, cache: ResponseCache = None):
        super().__init__(config)
        self.cache = cache or ResponseCache()
    
    async def create_message(self, **kwargs) -> Optional[Dict]:
        """Create message with caching"""
        # Generate cache key
        cache_key = self.cache.generate_key(**kwargs)
        
        # Try to get from cache
        cached_response = self.cache.get(cache_key)
        if cached_response:
            return cached_response
        
        # Call API
        response = await super().create_message(**kwargs)
        
        if response and self.cache:
            # Cache the response
            model = kwargs.get('model', 'unknown')
            tokens = response.get('usage', {}).get('total_tokens', 0)
            self.cache.set(cache_key, response, model, tokens)
        
        return response

# Test caching system
async def test_caching():
    config = APIConfig(api_key=userdata.get('ANTHROPIC_API_KEY'))
    cache = ResponseCache()
    
    async with CachedClaudeClient(config, cache) as client:
        # First call (cache miss)
        print("First call (cache miss)...")
        start = time.time()
        response1 = await client.create_message(
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "What is machine learning?"}]
        )
        time1 = time.time() - start
        print(f"Time taken: {time1:.2f}s")
        
        # Second call (cache hit)
        print("\nSecond call (cache hit)...")
        start = time.time()
        response2 = await client.create_message(
            model="claude-3-haiku-20240307",
            messages=[{"role": "user", "content": "What is machine learning?"}]
        )
        time2 = time.time() - start
        print(f"Time taken: {time2:.2f}s")
        print(f"Speedup: {time1/time2:.1f}x faster")
        
        # Show cache stats
        stats = cache.get_stats()
        print(f"\nCache stats: {stats}")

await test_caching()
```

#### **Step 3: Monitoring & Analytics**
```python
# Cell 3: Monitoring System
from collections import defaultdict
import statistics
from typing import Dict, List
import json
from datetime import datetime

class APIMonitor:
    """Monitor API usage and performance"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.errors = []
        self.start_time = datetime.now()
    
    def record_request(self, 
                      model: str, 
                      duration: float, 
                      tokens: int, 
                      success: bool):
        """Record API request metrics"""
        timestamp = datetime.now()
        
        self.metrics['requests'].append({
            'timestamp': timestamp,
            'model': model,
            'duration': duration,
            'tokens': tokens,
            'success': success
        })
        
        # Track per-model metrics
        self.metrics[f'model_{model}'].append({
            'duration': duration,
            'tokens': tokens,
            'success': success
        })
    
    def record_error(self, model: str, error: str, context: Dict = None):
        """Record API errors"""
        self.errors.append({
            'timestamp': datetime.now(),
            'model': model,
            'error': error,
            'context': context
        })
    
    def get_summary(self) -> Dict:
        """Get summary of API usage"""
        if not self.metrics.get('requests'):
            return {"message": "No requests recorded"}
        
        requests = self.metrics['requests']
        total_requests = len(requests)
        successful = sum(1 for r in requests if r['success'])
        failed = total_requests - successful
        
        durations = [r['duration'] for r in requests]
        total_tokens = sum(r['tokens'] for r in requests)
        
        # Calculate costs (example pricing)
        # Note: Update with actual Claude pricing
        input_cost_per_million = 0.80  # Example: $0.80 per million input tokens
        output_cost_per_million = 4.00  # Example: $4.00 per million output tokens
        
        # Estimate costs (simplified)
        estimated_cost = (total_tokens / 1_000_000) * input_cost_per_million
        
        return {
            'total_requests': total_requests,
            'successful_requests': successful,
            'failed_requests': failed,
            'success_rate': successful / total_requests if total_requests > 0 else 0,
            'total_tokens': total_tokens,
            'avg_duration': statistics.mean(durations) if durations else 0,
            'min_duration': min(durations) if durations else 0,
            'max_duration': max(durations) if durations else 0,
            'estimated_cost': estimated_cost,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'requests_per_hour': total_requests / ((datetime.now() - self.start_time).total_seconds() / 3600) if total_requests > 0 else 0
        }
    
    def get_model_breakdown(self) -> Dict:
        """Get breakdown by model"""
        breakdown = {}
        
        for key in self.metrics:
            if key.startswith('model_'):
                model = key.replace('model_', '')
                metrics = self.metrics[key]
                
                if metrics:
                    durations = [m['duration'] for m in metrics]
                    tokens = sum(m['tokens'] for m in metrics)
                    successful = sum(1 for m in metrics if m['success'])
                    
                    breakdown[model] = {
                        'requests': len(metrics),
                        'successful': successful,
                        'tokens': tokens,
                        'avg_duration': statistics.mean(durations) if durations else 0,
                        'success_rate': successful / len(metrics) if len(metrics) > 0 else 0
                    }
        
        return breakdown
    
    def generate_report(self) -> str:
        """Generate monitoring report"""
        summary = self.get_summary()
        breakdown = self.get_model_breakdown()
        
        report = f"""
        📊 API Monitoring Report
        {'='*50}
        
        Summary:
        - Total Requests: {summary['total_requests']}
        - Success Rate: {summary['success_rate']:.1%}
        - Total Tokens: {summary['total_tokens']:,}
        - Estimated Cost: ${summary['estimated_cost']:.4f}
        - Avg Response Time: {summary['avg_duration']:.2f}s
        - Uptime: {summary['uptime_hours']:.1f} hours
        
        Model Breakdown:
        """
        
        for model, stats in breakdown.items():
            report += f"""
            {model}:
              - Requests: {stats['requests']}
              - Success Rate: {stats['success_rate']:.1%}
              - Tokens: {stats['tokens']:,}
              - Avg Duration: {stats['avg_duration']:.2f}s
            """
        
        if self.errors:
            report += f"""
            
            Recent Errors ({len(self.errors)}):
            """
            for error in self.errors[-5:]:  # Last 5 errors
                report += f"""
            - {error['timestamp'].strftime('%H:%M:%S')}: {error['model']} - {error['error'][:100]}
                """
        
        return report
    
    def save_to_file(self, filename: str = "monitoring_report.json"):
        """Save monitoring data to file"""
        data = {
            'summary': self.get_summary(),
            'model_breakdown': self.get_model_breakdown(),
            'errors': self.errors[-100:],  # Last 100 errors
            'generated_at': datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"✅ Monitoring report saved to {filename}")

# Integrated client with monitoring
class MonitoredClaudeClient(CachedClaudeClient):
    """Claude client with monitoring"""
    
    def __init__(self, config: APIConfig, cache: ResponseCache = None):
        super().__init__(config, cache)
        self.monitor = APIMonitor()
    
    async def create_message(self, **kwargs) -> Optional[Dict]:
        """Create message with monitoring"""
        model = kwargs.get('model', 'unknown')
        start_time = time.time()
        
        try:
            response = await super().create_message(**kwargs)
            duration = time.time() - start_time
            
            if response:
                tokens = response.get('usage', {}).get('total_tokens', 0)
                self.monitor.record_request(model, duration, tokens, True)
            else:
                self.monitor.record_request(model, duration, 0, False)
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            self.monitor.record_request(model, duration, 0, False)
            self.monitor.record_error(model, str(e), kwargs)
            raise

# Test monitoring
async def test_monitoring():
    config = APIConfig(api_key=userdata.get('ANTHROPIC_API_KEY'))
    
    async with MonitoredClaudeClient(config) as client:
        # Make several requests
        for i in range(3):
            try:
                response = await client.create_message(
                    model="claude-3-haiku-20240307",
                    messages=[{"role": "user", "content": f"Test message {i}"}],
                    max_tokens=50
                )
                print(f"Request {i+1}: Success")
            except Exception as e:
                print(f"Request {i+1}: Failed - {e}")
        
        # Generate report
        report = client.monitor.generate_report()
        print(report)
        
        # Save report
        client.monitor.save_to_file()

await test_monitoring()
```

#### **Step 4: Async Task Queue System**
```python
# Cell 4: Task Queue System
import asyncio
from asyncio import Queue
from typing import Callable, Any
import uuid
from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """API Task definition"""
    id: str
    model: str
    messages: List[Dict]
    max_tokens: int
    temperature: float
    system: Optional[str]
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: datetime = None
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class TaskQueue:
    """Async task queue for API requests"""
    
    def __init__(self, 
                 client: MonitoredClaudeClient,
                 max_workers: int = 3,
                 max_queue_size: int = 100):
        self.client = client
        self.max_workers = max_workers
        self.queue = Queue(maxsize=max_queue_size)
        self.tasks: Dict[str, Task] = {}
        self.workers = []
        self.is_running = False
    
    async def start(self):
        """Start task queue workers"""
        self.is_running = True
        self.workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.max_workers)
        ]
        print(f"🚀 Started {self.max_workers} worker(s)")
    
    async def stop(self):
        """Stop task queue workers"""
        self.is_running = False
        # Wait for all workers to complete
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        print("🛑 Task queue stopped")
    
    async def _worker(self, worker_id: int):
        """Worker process for handling tasks"""
        print(f"👷 Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get task from queue
                task = await self.queue.get()
                
                if task is None:  # Sentinel for shutdown
                    break
                
                # Update task status
                task.status = TaskStatus.PROCESSING
                
                try:
                    # Execute API call
                    result = await self.client.create_message(
                        model=task.model,
                        messages=task.messages,
                        max_tokens=task.max_tokens,
                        temperature=task.temperature,
                        system=task.system
                    )
                    
                    # Update task with result
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                    task.completed_at = datetime.now()
                    
                except Exception as e:
                    # Update task with error
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = datetime.now()
                
                finally:
                    # Mark task as done
                    self.queue.task_done()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker {worker_id} error: {e}")
                continue
    
    async def submit_task(self, 
                         model: str,
                         messages: List[Dict],
                         max_tokens: int = 1000,
                         temperature: float = 0.7,
                         system: str = None) -> str:
        """Submit a new task to the queue"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system
        )
        
        # Add to tracking
        self.tasks[task_id] = task
        
        # Add to queue
        await self.queue.put(task)
        
        print(f"📝 Submitted task {task_id}")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get status of a task"""
        return self.tasks.get(task_id)
    
    def get_stats(self) -> Dict:
        """Get queue statistics"""
        pending = sum(1 for t in self.tasks.values() 
                     if t.status == TaskStatus.PENDING)
        processing = sum(1 for t in self.tasks.values() 
                        if t.status == TaskStatus.PROCESSING)
        completed = sum(1 for t in self.tasks.values() 
                       if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks.values() 
                    if t.status == TaskStatus.FAILED)
        
        return {
            'total_tasks': len(self.tasks),
            'pending': pending,
            'processing': processing,
            'completed': completed,
            'failed': failed,
            'success_rate': completed / len(self.tasks) if len(self.tasks) > 0 else 0,
            'queue_size': self.queue.qsize()
        }

# Example: Batch processing with task queue
async def example_task_queue():
    config = APIConfig(api_key=userdata.get('ANTHROPIC_API_KEY'))
    cache = ResponseCache()
    
    async with MonitoredClaudeClient(config, cache) as client:
        # Create task queue
        task_queue = TaskQueue(client, max_workers=2)
        
        # Start queue
        await task_queue.start()
        
        # Submit multiple tasks
        task_ids = []
        for i in range(5):
            task_id = await task_queue.submit_task(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": f"Explain topic {i} in simple terms"}],
                max_tokens=200
            )
            task_ids.append(task_id)
        
        # Monitor task progress
        print("\n📊 Monitoring task progress...")
        for _ in range(10):  # Check 10 times
            stats = task_queue.get_stats()
            print(f"Queue stats: {stats}")
            
            if stats['completed'] + stats['failed'] == len(task_ids):
                print("✅ All tasks completed!")
                break
            
            await asyncio.sleep(1)
        
        # Get results
        print("\n📋 Task Results:")
        for task_id in task_ids:
            task = task_queue.get_task_status(task_id)
            if task:
                print(f"\nTask {task_id}: {task.status.value}")
                if task.result:
                    print(f"  Response: {task.result.get('content', [{}])[0].get('text', '')[:100]}...")
                if task.error:
                    print(f"  Error: {task.error}")
        
        # Show final stats
        final_stats = task_queue.get_stats()
        print(f"\n📈 Final Stats: {final_stats}")
        
        # Stop queue
        await task_queue.stop()

await example_task_queue()
```

#### **Step 5: Complete Production Application**
```python
# Cell 5: Complete Production Application Template
"""
Complete Production Claude API Application Template
Save this as: production_claude_app.py
"""

import asyncio
import aiohttp
import json
import time
import sqlite3
import hashlib
import pickle
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
from collections import defaultdict
from tenacity import retry, stop_after_attempt, wait_exponential
from ratelimit import limits, sleep_and_retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== DATA CLASSES ====================
@dataclass
class APIConfig:
    api_key: str
    base_url: str = "https://api.anthropic.com/v1"
    max_retries: int = 3
    timeout: int = 30
    rate_limit_per_minute: int = 50

@dataclass
class Task:
    id: str
    model: str
    messages: List[Dict]
    max_tokens: int
    temperature: float
    system: Optional[str] = None
    status: str = "pending"
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: datetime = None
    completed_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

# ==================== CACHE SYSTEM ====================
class ResponseCache:
    def __init__(self, cache_dir: str = "./cache", ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
        self.db_path = self.cache_dir / "responses.db"
        self._init_database()
    
    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    response BLOB,
                    model TEXT,
                    tokens INTEGER,
                    created_at TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
    
    def get(self, key: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT response FROM cache WHERE key = ? AND expires_at > ?",
                (key, datetime.now())
            )
            row = cursor.fetchone()
            return pickle.loads(row[0]) if row else None
    
    def set(self, key: str, response: Dict, model: str, tokens: int):
        expires_at = datetime.now() + self.ttl
        response_blob = pickle.dumps(response)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO cache 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (key, response_blob, model, tokens, datetime.now(), expires_at))

# ==================== MONITORING ====================
class APIMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_time = datetime.now()
    
    def record_request(self, model: str, duration: float, tokens: int, success: bool):
        self.metrics['requests'].append({
            'timestamp': datetime.now(),
            'model': model,
            'duration': duration,
            'tokens': tokens,
            'success': success
        })
    
    def get_summary(self) -> Dict:
        requests = self.metrics.get('requests', [])
        if not requests:
            return {}
        
        total = len(requests)
        successful = sum(1 for r in requests if r['success'])
        total_tokens = sum(r['tokens'] for r in requests)
        
        return {
            'total_requests': total,
            'success_rate': successful / total if total > 0 else 0,
            'total_tokens': total_tokens,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }

# ==================== API CLIENT ====================
class ProductionClaudeClient:
    def __init__(self, config: APIConfig):
        self.config = config
        self.session = None
        self.cache = ResponseCache()
        self.monitor = APIMonitor()
        self.request_timestamps = []
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={
                "x-api-key": self.config.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def _check_rate_limit(self):
        now = time.time()
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if now - ts < 60
        ]
        
        if len(self.request_timestamps) >= self.config.rate_limit_per_minute:
            sleep_time = 60 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        self.request_timestamps.append(now)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_request(self, endpoint: str, data: Dict) -> Optional[Dict]:
        self._check_rate_limit()
        
        url = f"{self.config.base_url}/{endpoint}"
        
        try:
            async with self.session.post(url, json=data) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    retry_after = int(response.headers.get('retry-after', 5))
                    await asyncio.sleep(retry_after)
                    raise Exception("Rate limited")
                else:
                    error_text = await response.text()
                    logger.error(f"API error {response.status}: {error_text}")
                    return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            raise
    
    async def create_message(self, 
                           model: str = "claude-3-sonnet-20240229",
                           messages: List[Dict] = None,
                           max_tokens: int = 1000,
                           temperature: float = 0.7,
                           system: str = None,
                           use_cache: bool = True) -> Optional[Dict]:
        
        # Generate cache key
        cache_key_data = {
            "model": model,
            "messages": messages or [],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system
        }
        cache_key = hashlib.sha256(
            json.dumps(cache_key_data, sort_keys=True).encode()
        ).hexdigest()
        
        # Try cache
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                logger.info(f"Cache hit for {cache_key[:16]}...")
                return cached
        
        # API call
        start_time = time.time()
        
        data = {
            "model": model,
            "messages": messages or [],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        if system:
            data["system"] = system
        
        response = await self._make_request("messages", data)
        
        # Record metrics
        duration = time.time() - start_time
        tokens = response.get('usage', {}).get('total_tokens', 0) if response else 0
        success = response is not None
        
        self.monitor.record_request(model, duration, tokens, success)
        
        # Cache response
        if response and use_cache:
            self.cache.set(cache_key, response, model, tokens)
        
        return response
    
    async def batch_process(self, 
                          requests: List[Dict],
                          max_concurrent: int = 5) -> List[Optional[Dict]]:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(request):
            async with semaphore:
                return await self.create_message(**request)
        
        tasks = [process_with_semaphore(req) for req in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_monitoring_summary(self) -> Dict:
        return self.monitor.get_summary()

# ==================== TASK QUEUE ====================
class TaskQueue:
    def __init__(self, client: ProductionClaudeClient, max_workers: int = 3):
        self.client = client
        self.max_workers = max_workers
        self.queue = asyncio.Queue()
        self.tasks: Dict[str, Task] = {}
        self.workers = []
        self.is_running = False
    
    async def start(self):
        self.is_running = True
        self.workers = [
            asyncio.create_task(self._worker(i))
            for i in range(self.max_workers)
        ]
        logger.info(f"Started {self.max_workers} worker(s)")
    
    async def stop(self):
        self.is_running = False
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Task queue stopped")
    
    async def _worker(self, worker_id: int):
        while self.is_running:
            try:
                task = await self.queue.get()
                if task is None:
                    break
                
                task.status = "processing"
                
                try:
                    result = await self.client.create_message(
                        model=task.model,
                        messages=task.messages,
                        max_tokens=task.max_tokens,
                        temperature=task.temperature,
                        system=task.system
                    )
                    
                    task.status = "completed"
                    task.result = result
                    task.completed_at = datetime.now()
                    
                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                    task.completed_at = datetime.now()
                
                finally:
                    self.queue.task_done()
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                continue
    
    async def submit_task(self, **kwargs) -> str:
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            model=kwargs.get('model', 'claude-3-haiku-20240307'),
            messages=kwargs.get('messages', []),
            max_tokens=kwargs.get('max_tokens', 1000),
            temperature=kwargs.get('temperature', 0.7),
            system=kwargs.get('system')
        )
        
        self.tasks[task_id] = task
        await self.queue.put(task)
        
        logger.info(f"Submitted task {task_id}")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        return self.tasks.get(task_id)

# ==================== EXAMPLE USAGE ====================
async def main():
    """Example usage of the production system"""
    
    # Configuration
    config = APIConfig(
        api_key="your-api-key-here"  # Replace with actual key
    )
    
    # Initialize client
    async with ProductionClaudeClient(config) as client:
        # Create task queue
        task_queue = TaskQueue(client, max_workers=2)
        await task_queue.start()
        
        # Submit tasks
        task_ids = []
        for i in range(3):
            task_id = await task_queue.submit_task(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": f"Explain concept {i}"}],
                max_tokens=200
            )
            task_ids.append(task_id)
        
        # Wait for completion
        await asyncio.sleep(5)
        
        # Get results
        for task_id in task_ids:
            task = task_queue.get_task(task_id)
            if task:
                print(f"\nTask {task_id}: {task.status}")
                if task.result:
                    print(f"Response: {task.result.get('content', [{}])[0].get('text', '')[:100]}...")
        
        # Show monitoring summary
        summary = client.get_monitoring_summary()
        print(f"\n📊 Monitoring Summary: {summary}")
        
        # Cleanup
        await task_queue.stop()

# ==================== RUN IN COLAB ====================
if __name__ == "__main__":
    # For Colab, we need to run async code differently
    try:
        asyncio.run(main())
    except RuntimeError:
        # Colab may already have a running event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a task instead
            asyncio.create_task(main())
        else:
            loop.run_until_complete(main())

print("✅ Production application template ready!")
print("📥 To download: files.download('production_claude_app.py')")
```

### **Use Cases:**
1. **High-Volume Applications**: Handle thousands of API requests
2. **Cost Optimization**: Reduce API costs with caching
3. **Performance Monitoring**: Track API performance and errors
4. **Reliable Systems**: Build fault-tolerant applications
5. **Enterprise Integration**: Integrate Claude into existing systems

---

## **Deployment from Google Colab:**

### **Export Applications:**
```python
# Export complete applications
applications = {
    'claude_cli_app.py': cli_code,
    'streamlit_app.py': streamlit_code,
    'data_analysis_app.py': analysis_code,
    'production_app.py': production_code
}

for filename, code in applications.items():
    with open(filename, 'w') as f:
        f.write(code)
    print(f"✅ Saved: {filename}")

# Download all files
from google.colab import files
for filename in applications.keys():
    files.download(filename)
```

### **Create Requirements File:**
```python
requirements = """
anthropic>=0.8.0
aiohttp>=3.9.0
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
plotly>=5.17.0
tenacity>=8.2.0
python-dotenv>=1.0.0
rich>=13.0.0
sqlite3
"""

with open('requirements.txt', 'w') as f:
    f.write(requirements)

files.download('requirements.txt')
```

This comprehensive Claude AI lab series provides students with practical experience building production-ready applications, from basic API calls to sophisticated systems with caching, monitoring, and async processing.
