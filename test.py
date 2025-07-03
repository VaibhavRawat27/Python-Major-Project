import requests

GEMINI_API_KEY = "AIzaSyDye2pJRyzEqd_TMLYdj4f37d038buoLC8"
text = "Summarize this: This event was well organized. The food was average. The host was great but the location was too far."

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={GEMINI_API_KEY}"
headers = {"Content-Type": "application/json"}
data = {
    "contents": [{"parts": [{"text": text}]}]
}

res = requests.post(url, headers=headers, json=data)
print(res.status_code)
print(res.text)
