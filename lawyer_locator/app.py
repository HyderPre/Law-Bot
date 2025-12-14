from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)

def get_coordinates(address):
    """Get latitude & longitude from address using Nominatim (OpenStreetMap)."""
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json"}
    headers = {"User-Agent": "lawyer-locator-app"}
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    if data:
        return float(data[0]["lat"]), float(data[0]["lon"])
    return None, None


def get_nearby_lawyers(lat, lon):
    """Use Overpass API to find lawyers / affidavit makers near given lat, lon."""
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json];
    (
      node["office"="lawyer"](around:5000,{lat},{lon});
      node["office"="notary"](around:5000,{lat},{lon});
      node["service"="affidavit"](around:5000,{lat},{lon});
    );
    out center;
    """
    headers = {"User-Agent": "lawyer-locator-app"}
    response = requests.post(overpass_url, data={"data": query}, headers=headers)
    data = response.json()
    results = []
    for element in data.get("elements", []):
        name = element.get("tags", {}).get("name", "Unnamed Office")
        lat = element["lat"]
        lon = element["lon"]
        addr = element.get("tags", {}).get("addr:full", "Address not available")
        results.append({"name": name, "lat": lat, "lon": lon, "address": addr})
    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    address = data.get("address")
    lat, lon = get_coordinates(address)
    if not lat or not lon:
        return jsonify({"error": "Address not found."}), 404

    places = get_nearby_lawyers(lat, lon)
    return jsonify({"lat": lat, "lon": lon, "places": places})


if __name__ == "__main__":
    app.run(debug=True)
