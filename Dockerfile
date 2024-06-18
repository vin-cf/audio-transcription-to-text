# Use an official Python runtime as a parent image
FROM python:3.12-slim as build

# Set the working directory in the container
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container
COPY . /app

# Stage 2: Runtime
# slim, not alpine is necessary for shared libraries libgcc_s.so.1, ld-linux-x86-64.so.2
FROM python:3.12-slim

EXPOSE 5002

WORKDIR /app

# Copy dependencies and application code from the build stage
COPY --from=build /root/.local /root/.local
COPY --from=build /app /app

ENV PATH=/root/.local/bin:$PATH

CMD ["python", "main.py"]
