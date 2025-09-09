# Running Code on a GPU

## Setting up the environment
```bash
git clone https://github.com/Harsheys18/CS791_Assignment_3.git
cd CS791_Assignment_3
python3 -m venv virtual-env
source virtual-env/bin/activate
pip install -r requirements.txt
```

My token:
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
Username:
Harsheys18

# Setting Up a GitHub Personal Access Token (PAT)

To create a personal access token in GitHub, follow these steps:

1. **Log in to GitHub**  
   Go to [github.com](https://github.com) and log in with your account.

2. **Navigate to Settings**  
   In the upper-right corner of any GitHub page, click your profile picture and select **Settings**.

3. **Access Developer Settings**  
   In the left sidebar of the settings page, click **Developer settings**.

4. **Go to Personal Access Tokens**  
   Under **Developer settings**, click **Personal access tokens**.

5. **Choose Token Type**  
   You will see options for:  
   - **Tokens (classic)**  
   - **Fine-grained tokens**  

   Choose the type that best suits your needs. For general use, **Tokens (classic)** is often sufficient.

6. **Generate a New Token**  
   Click **Generate new token** (or **Generate new token (classic)** if you selected classic tokens) and follow the prompts to configure permissions and expiration.
   Save this token, and paste after running the "Setting up the environment commands".

### following instructions can be ignored for now.

# Environment and installations

To install Miniconda, follow the steps [here](https://www.anaconda.com/docs/getting-started/miniconda/install).

To setup the environment, follow these steps:

```
conda create --name cs791env python=3.8 -y
conda activate cs791env
```

Install the dependencies (if any):
```
pip install -r requirements.txt
```

To install torch, you can follow the steps [here](https://pytorch.org/get-started/locally/). You'll need to know the cuda version on the server. Use `nvitop` command to know the version first. If you have cuda version 12.4, you can just do:
```
pip install torch
```

To check if GPU is connected, run the command.
```
print("CUDA available:", torch.cuda.is_available())
```

In case multiple GPUs are present in the system, we recommend using the environment variable `CUDA_VISIBLE_DEVICES` when running your scripts. For example, below command ensures that your script runs on 7th GPU. 
```
CUDA_VISIBLE_DEVICES=7 python d3pm_template.py --mode train
```

CUDA error messages can often be cryptic and difficult to debug. In such cases, the following command can be quite useful:
```
CUDA_VISIBLE_DEVICE=-1 python d3pm_template.py --mode train
```
This forces the script to run exclusively on the CPU.



# Server Connection

1. In VS Code, install Remote – SSH extension. Make sure OpenSSH client is installed on your system.

2. Open Remote Explorer in VS Code.

Click "+" and select Add New SSH Host.
Enter server details in the format:
username@hostname
Example:
alice@gpu1.cse.iitb.ac.in

OR,

edit ~/.ssh/config directly:
Host gpu1.cse.iitb.ac.in
    HostName gpu1.cse.iitb.ac.in
    User alice

3. In Remote Explorer, click your saved server (e.g. server1). VS Code will prompt for the password. Once authenticated, a new VS Code window opens connected to the server.

For more details, visit, https://code.visualstudio.com/docs/remote/ssh.

To forcefully shutdown kernel, use these commands:
```
import os
os._exit(0)
```


