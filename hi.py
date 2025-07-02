import requests

url = "https://www.searchapi.io/api/v1/search"
params = {
  "engine": "google_shopping",
  "q": "PS5",
  "gl": "us",
  "hl": "en",
  "location": "California,United States",
  "api_key": "f4XXBhvMAPP3XN3C3UrhyptY"
}

response = requests.get(url, params=params)
print(response.text)