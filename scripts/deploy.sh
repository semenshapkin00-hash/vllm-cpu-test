#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MINIKUBE=${MINIKUBE:-minikube}
profile=${PROFILE:-vllm-cpu}
if ! "$MINIKUBE" -p "$profile" status >/dev/null 2>&1; then
  "$MINIKUBE" start --profile "$profile" --driver=docker --cpus=6 --memory=6000mb \
    --disk-size=16g --kubernetes-version=v1.34.0
fi
"$MINIKUBE" -p "$profile" image load vllm-cpu-api:0.27.0
"$MINIKUBE" -p "$profile" image load vllm-model-init:0.27.0
kubectl --context "$profile" apply -f k8s/namespace.yaml
kubectl --context "$profile" -n vllm-cpu create secret generic vllm-secrets \
  --from-file=api-token=.secrets/api-token --from-file=hf-token=.secrets/hf-token \
  --dry-run=client -o yaml | kubectl --context "$profile" apply -f -
kubectl --context "$profile" apply -k k8s
kubectl --context "$profile" -n vllm-cpu rollout status deployment/vllm --timeout=15m
