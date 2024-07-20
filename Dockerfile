# Use an official Python runtime as a parent image
FROM python:3.12-slim AS build

# Set the working directory in the container
WORKDIR /app
COPY pyproject.toml pyproject.toml
COPY . .
RUN python -m venv /.venv
RUN /.venv/bin/python -m pip install -U setuptools wheel
RUN /.venv/bin/pip install -q .

# Install any needed packages specified in requirements.txt
RUN /.venv/bin/pip install .

# Stage 2: Runtime
# slim, not alpine is necessary for shared libraries libgcc_s.so.1, ld-linux-x86-64.so.2
FROM python:3.12-slim

EXPOSE 5002

WORKDIR /app

# TODO: Copy offline transformer models

# Copy dependencies and application code from the build stage
#COPY --from=build /root/.local /root/.local
COPY --from=build /app /app
COPY --from=build /.venv /.venv

ENV PATH=/.venv/bin:$PATH

CMD ["python", "main.py"]
