FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agentb/ agentb/
COPY agentb.yaml.example agentb.yaml

EXPOSE 50001

CMD ["python", "-m", "agentb.server"]
