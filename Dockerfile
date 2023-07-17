# This is a potassium-standard dockerfile, compatible with Banana

# Don't change this. Currently we only support this specific base image.
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
# TODO add the missing dependencies
RUN apt-get update && apt-get install -y git wget

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Download the weights from a url
# TODO: replace the url with your own
RUN wget http://www.example.com/weights.pt 

ADD . .

EXPOSE 8000

CMD python3 -u app.py