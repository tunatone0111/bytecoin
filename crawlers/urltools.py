from urllib.parse import urlparse, parse_qs

def qdic(url):
  parts = urlparse(url)
  return parse_qs(parts.query)

def replace_query_string(url, key, value):
  qs = dict(parse_qsl(parts.query))