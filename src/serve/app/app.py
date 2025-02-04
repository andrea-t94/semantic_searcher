from flask import Flask, request, jsonify

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search

app = Flask(__name__)

@app.route('/find_similar', methods=['POST'])
def find_similar():
    try:
        data = request.get_json()

        if not data or 'query' not in data or 'documents' not in data:
            return jsonify({'error': 'Missing query or documents in request'}), 400

        query = data['query']
        documents = data['documents']
        if not isinstance(documents, list) or not documents:
            return jsonify({'error': 'Documents must be a non-empty list'}), 400

        # Encode query and documents
        query_embedding = model.encode(query)
        doc_embeddings = model.encode(documents)

        # Compute similarities
        results = semantic_search(query_embedding, doc_embeddings, top_k=1)

        response = {
            'most_similar_doc': documents[int(results[0][0]['corpus_id'])],
            'similarity_score': results[0][0]['score'],
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    try:
        model = SentenceTransformer(BASELINE_MODEL_PATH)
    except:
        print(f"No local model found, downloading baseline from hub")
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-cos-v5')
    app.run(host='0.0.0.0', port=5000)