from urllib.parse import urlparse, parse_qs


def get_query(url):
    """Parses the query part of an url to dictionaries.
    ex) 

    Args:
        url (string): url

    Returns:
        dict: query dictionary
    """
    parts = urlparse(url)
    return parse_qs(parts.query)
