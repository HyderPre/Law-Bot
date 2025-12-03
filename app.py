from flask import Flask, request, jsonify
from flask_cors import CORS

# import your pipeline function
# if your file is named lawbot_pipeline.py and process_query is inside it:
from lawbot import process_query  
# OR if your file is still named main_category_subcategory.py:
# from main_category_subcategory import process_query

app = Flask(__name__)
CORS(app)  # allow calls from your HTML file

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json() or {}
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # Your existing pipeline:
    # category, subtopic_id, matched_example, laws, advice = process_query(query)
    result = process_query(query)

    # in your latest version process_query returns:
    # category, subtopic_id, matched_example, laws, advice
    if len(result) == 5:
        category, subtopic_id, matched_example, laws, advice = result
        reference_cases = []  # we'll fill later
    elif len(result) == 6:
        category, subtopic_id, matched_example, laws, advice, reference_cases = result
    else:
        return jsonify({"error": "Unexpected pipeline response format"}), 500

    # you may want a nicer subtopic title later; for now we send id
    response = {
        "category": category,
        "subtopic_id": subtopic_id,
        "subtopic_title": subtopic_id,  # change later if you have a title
        "laws": laws or [],
        "advice": advice or "",
        "reference_cases": reference_cases or [],
    }
    return jsonify(response)

if __name__ == "__main__":
    # run on http://127.0.0.1:5000
    app.run(debug=True)
