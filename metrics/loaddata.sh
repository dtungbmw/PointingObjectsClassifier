curl -X POST https://localhost:9200/_bulk?pretty \
          -H "Authorization: ApiKey 0hNIOTgZS9upvaXClICjuQ" \
            -H "Content-Type: application/json" \
              -d'
              { "index" : { "_index" : "ml_confusion_matrix" } }
              {"name": "foo", "title": "bar" }'
