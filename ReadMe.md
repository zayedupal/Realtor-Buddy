-- 1. If you want to run locally on your local machine. Create a virtual env first & then run 


pip install -r requirements.txt or


-- 2. If you want to build a docker image & run

docker build -t gemini_app_local:latest -f Dockerfile .

docker run -it -p 8501:8501 gemini_app_local:latest


-- 3. If you need to deploy a docker image to Google Cloud Run from Mac M1/M2 machines

docker buildx build --platform linux/amd64 -t gemini_app .