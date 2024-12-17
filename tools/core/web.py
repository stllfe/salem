from typing import Literal

from tools.runtime import CURRENT


Language = Literal["ru", "en"]


def search_topk(query: str, k: int = 5) -> list[dict[str, str]]:
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


def get_page_content(url: str) -> str:
  """Extract the readable content from the given page URL in the Markdown format.

  Args:
    url: A string starting with 'http://' or 'https://' or '@' and a short uid
      to get the URL from the last `search_topk` call (e.g. '@4x34b')

  Returns:
    A markdown string with the page content
  """


# use-case for the Standford's WikiChat:
# https://search.genie.stanford.edu/redoc
def search_wiki(query: str, k: int = 5, language: Language = CURRENT.LANGUAGE) -> list[dict[str, str]]:
  """Get top K search results from Wikipedia for the given query.

  Note: it returns short relevant extracts (not full pages) if found.

  Args:
    query: The search query in a natural form
    k: The amount of top relevant extracts to return

  Returns:
    A list of K dictionaries with URLs and short Wikipedia extracts as well as some metadata.
  """
