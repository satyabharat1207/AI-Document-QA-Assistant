#stage 1 : I need base image
from python:3.11-slim

#stage 2 : I need a floder to store my code in dockerfile
WORKDIR /app

#stage 3 : Copy the dependency file
copy requirements.txt .

#stage 4 : Run the dependency file
run pip install --upgrade pip
run pip install -r requirements.txt

#stage 5 : Copy the entire code
copy . .

#stage 6 : Expose the ports
expose 8501

#stage 7 : Run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

