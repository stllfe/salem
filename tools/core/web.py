from typing import Any

from tools.core.backend.web import Browser
from tools.runtime import runtime


browser = runtime.get_tool(Browser)


def search_topk(query: str, k: int = 3) -> list[dict[str, Any]]:
  """Get top K search results for the given query (just like a desktop browser with a search engine).

  Note: this only returns the URLs and their short descriptions.
    Use `get_page_content` to get the actual page content in a readable format!

  Args:
    query: The search query in a natural form
    k: The amount of top relevant pages to return

  Returns:
    A list of K dictionaries with URLs, their descriptions and short unique IDs
    for referencing in answers or other functions
  """

  return [link.dump() for link in browser.search_topk(query, k=k)]


def get_page_content(url: str) -> str:
  """Extract the readable content from the given page URL in the Markdown format.
  Use this to browse webpages.

  Args:
    url: A string starting with 'http://' or 'https://' or '@' and a short uid
      to get the URL from the last `search_topk` call (e.g. '@4x34b')

  Returns:
    A markdown string with the page content
  """

  return browser.get_page_content(url)


def search_wiki(query: str, k: int = 5) -> list[dict[str, str]]:
  """Get top K search results from Wikipedia for the given query.

  Note: it returns short relevant extracts (not full pages) if found.

  Args:
    query: The search query in a natural form
    k: The amount of top relevant extracts to return

  Returns:
    A list of K dictionaries with URLs and short Wikipedia extracts as well as some metadata.
  """

  return [link.dump() for link in browser.search_wiki(query, k=k)]
