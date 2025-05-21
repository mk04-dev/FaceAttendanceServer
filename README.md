# FaceAttendanceServer

<details open>
<summary>Install</summary>

```sh
    # Clone the YOLOv5 repository
    git clone https://github.com/mk04-dev/FaceAttendanceServer.git

    # Navigate to the cloned directory
    cd FaceAttendanceServer

    # Create environment
    python -m venv .env

    # Active environment
    ## Windows
    ./.env/Scripts/activate

    ## Linux
    source ./.env/bin/activate

    # Install required packages
    pip install -r requirements.txt
```

</details>

```sh
    # Run Redis server
    ## Windows
    ./Redis/redis-server.exe ./Redis/redis.windows.conf
    ## Linux
    ./Redis/redis-server.exe

    # Run server
    ./.env/Scripts/activate

    python main.py
```

</details>
