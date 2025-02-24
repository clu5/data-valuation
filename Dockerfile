FROM nvidia/cuda:12.6.3-runtime-ubuntu22.04


# Set the working directory in the container
WORKDIR /app

# Set up Python environment
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    python3.11-venv \
    # Your other dependencies
    build-essential \
    curl \
    g++ \
    gcc \
    git \
    gnupg \
    wget \ 
    vim \
    && rm -rf /var/lib/apt/lists/*


# Change the terminal prompt color
RUN echo 'PS1="\[\e[36m\]\u@\h:\w\$\[\e[m\] "' >> /root/.bashrc

# Configure vi bindings for the terminal
RUN echo "set editing-mode vi" >> /etc/inputrc

# Configure Vim history and undofile
RUN echo "set history=1000" >> /etc/vim/vimrc \
    && echo "set undofile" >> /etc/vim/vimrc

# Set the DATA_DIR environment variable
ENV DATA_DIR=/mnt/data

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Change to your actual UID from the system run `id -u`
ARG ID_UID=17649  

# Change to your actual username from the system run `whoami`
ARG ID_NAME=luchar

# Change to your actual group ID from the system run `id -gn`
ARG ID_GID=2000   

# Create the user and group
RUN groupadd -g ${ID_GID} mlusers && useradd -g ${ID_GID} -u ${ID_UID} -m -s /bin/bash ${ID_NAME}

# Expose the port the app runs on
# EXPOSE 8888

# Command to run the Jupyter notebook
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]