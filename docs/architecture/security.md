"""TLS and encryption notes.

Production: terminate HTTPS at a reverse proxy (Caddy/nginx) in front of FastAPI.
Development: HTTP localhost.

Sensitive fields in PostgreSQL should use application-level encryption in Phase 19.
Federated clients send weight updates over TLS; they never send raw CXR files to the aggregator.
"""
