curl -X POST \
  -H "Content-Type: application/json" \
  -H "x-customerkey: test" \
  -d '{"query": "What did paul work on?", "parent_ids": ["paul"]}' \
  172.17.0.3:8080/query