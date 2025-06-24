FROM nvcr.io/nvidia/tensorflow:24.05-tf2-py3

# 2. Set the working directory inside the container
WORKDIR /workspace

# 3. Copy and install Python requirements
# This step is cached, so it only runs when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your source code into the container's /workspace directory
COPY ./src ./src

# 5. [OPTIONAL] Set a default command that keeps the container running
# This command does nothing but prevents the container from exiting immediately.
# It's useful if you want a long-running container to attach to.
CMD ["bash"]