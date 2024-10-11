# opencv_timemeasurements

## Overview

`opencv_timemeasurements` is a Python-based application that uses OpenCV for video processing and Flask for web-based interaction. The application is designed to measure lap times of cars based on color detection in a video feed.

## Features

- **Real-time Video Processing**: Uses OpenCV to process video frames in real-time.
- **Color Detection**: Detects cars based on their color and measures lap times.
- **Web Interface**: Provides a web interface using Flask to start/stop the video feed and view lap times.
- **Configurable Settings**: Allows configuration of various parameters such as inactive time limit, contrast, brightness, and saturation.
- **Detailed Car View**: Displays detailed information for each tracked car, including all lap times, average lap time, and average speed.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/fiveBenilu/opencv_timemeasurements.git
    cd opencv_timemeasurements
    ```

2. **Create a virtual environment**:
    ```sh
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. **Install dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Flask application**:
    ```sh
    python main.py
    ```

2. **Access the web interface**:
    Open a web browser and go to `http://127.0.0.1:5000`.

3. **Start the video feed**:
    Click on the "Start" button to begin the video feed and start measuring lap times.

4. **Stop the video feed**:
    Click on the "Stop" button to stop the video feed.

5. **View lap times**:
    Lap times and the status of each car can be viewed on the web interface.

6. **View detailed car information**:
    Click on a car's name to view detailed information, including all tracked lap times, average lap time, and average speed.

## Configuration

The application allows you to configure various settings through the `config.html` page:

- **Inactive Time Limit**: The time limit after which a car is considered inactive.
- **Contrast**: Adjust the contrast of the video feed.
- **Brightness**: Adjust the brightness of the video feed.
- **Saturation**: Adjust the saturation of the video feed.
- **Car Colors**: Add or modify the colors of the cars to be detected.

## API Endpoints

- **`GET /`**: Renders the main page.
- **`GET /start`**: Starts the video feed.
- **`GET /stop`**: Stops the video feed.
- **`GET /video_feed`**: Provides the video feed.
- **`GET /rundenzeiten`**: Returns the lap times and status of each car.
- **`GET /config`**: Renders the configuration page.
- **`GET /get_config`**: Returns the current configuration.
- **`POST /save_config`**: Saves the configuration.
- **`POST /add_car`**: Adds a new car with the specified color.
- **`GET /car/<auto_id>`**: Renders the detailed view for a specific car.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the GPL V3 License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenCV](https://opencv.org/)
- [Flask](https://flask.palletsprojects.com/)

## Contact

For any questions or suggestions, please open an issue or contact the repository owner.