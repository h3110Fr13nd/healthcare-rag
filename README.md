```bash
docker  run  --network=ollama-webui_default  -p 6333:6333 -p 6334:6334     -v $(pwd)/qdrant_storage:/qdrant/storage:z --name healthcare-rag-qdrant-store  -d   qdrant/qdrant
```