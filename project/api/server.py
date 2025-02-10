from api.routes import app

if __name__ == "__main__":
    # Start the Flask API server on port 5000.
    app.run(host="0.0.0.0", port=5000, debug=True)