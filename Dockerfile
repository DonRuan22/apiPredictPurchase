FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

#define the working directory of Docker container
WORKDIR /app 

#copy everything in ./actions directory (your custom actions code) to /app/actions in container
COPY ./ ./

# install dependencies
#RUN pip install -r requirements.txt

# command to run on container start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

RUN ls

EXPOSE 5055