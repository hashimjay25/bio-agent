# ── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install dependencies into a separate layer for cache efficiency.
COPY agent/requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Non-root user for security
RUN groupadd -r agentuser && useradd -r -g agentuser agentuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy agent source
COPY agent/ .

# Hosted agents on Foundry listen on port 8088
EXPOSE 8088

USER agentuser

# Default: HTTP server mode (no --cli flag)
CMD ["python", "main.py", "--server"]
