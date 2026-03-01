FROM openpolicyagent/opa:latest
COPY bundle /bundle
CMD ["run", "--server", "--addr", ":8181", "/bundle/rego", "/bundle/data"]
