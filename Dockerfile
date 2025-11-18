FROM python:3.12-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


WORKDIR /app

# Copy project files
COPY pyproject.toml .
COPY README.md .
COPY . .

# Install dependencies using uv
RUN uv sync --locked

EXPOSE 8080

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
