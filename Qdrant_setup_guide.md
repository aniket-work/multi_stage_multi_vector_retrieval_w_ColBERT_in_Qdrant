# Setting Up Qdrant Using Docker

This guide will walk you through the process of setting up Qdrant using Docker, which is the easiest way to get started.

## Step 1: Install Docker

1. Visit the official Docker website: [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. Download Docker Desktop for Windows.
3. Follow the installation wizard to install Docker Desktop.
4. Once installed, start Docker Desktop.

## Step 2: Verify Docker Installation

1. Open a command prompt or PowerShell.
2. Run the following command to check if Docker is installed correctly:

    ```sh
    docker --version
    ```

3. You should see the Docker version information.

## Step 3: Pull the Qdrant Docker Image

1. In the same command prompt or PowerShell, run:

    ```sh
    docker pull qdrant/qdrant
    ```

2. This will download the latest Qdrant image.

## Step 4: Run Qdrant Container

1. Create a directory for Qdrant data (e.g., `C:\qdrant_data`).
2. Run the following command to start the Qdrant container:

    ```sh
    docker run -p 6333:6333 -p 6334:6334 -v C:\qdrant_data:/qdrant/storage:z qdrant/qdrant
    ```

    This command:
    - Maps port 6333 (REST API) and 6334 (gRPC) from the container to your host.
    - Mounts the local directory `C:\qdrant_data` to the container's storage.

## Step 5: Verify Qdrant is Running

1. Open a web browser and go to: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
2. You should see the Qdrant dashboard, indicating that Qdrant is running successfully.

## Step 6: Update Your Python Code

1. Open your `config.py` file.
2. Ensure the `QDRANT_URL` is set to:

    ```python
    QDRANT_URL = "http://localhost:6333"
    ```

## Step 7: Run Your Python Script

1. Open a new command prompt or PowerShell window.
2. Navigate to your project directory.
3. Run your Python script:

    ```sh
    python main.py
    ```

## Troubleshooting

- If you encounter permission issues with Docker, make sure you're running the command prompt or PowerShell as an administrator.
- If the ports are already in use, you can change them in the Docker run command and update your `config.py` accordingly.
- Ensure that Docker Desktop is running before trying to start the Qdrant container.

## Additional Notes

- To stop the Qdrant container, you can use `Ctrl+C` in the terminal where it's running, or use the Docker Desktop interface.
- To start the container again later, you can use the Docker Desktop interface or run:

    ```sh
    docker start <container_name_or_id>
    ```

By following these steps, you should have a local Qdrant instance running and accessible to your Python script.