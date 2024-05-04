pip install -r requirements.txt or

-- If you want to build a docker image & run

docker build -t gemini_app_local:latest -f Dockerfile .

docker run -it -p 8501:8501 gemini_app_local:latest


-- If you need to deploy to Google Cloud Run from Mac M1/M2 machines

docker buildx build --platform linux/amd64 -t gemini_app .