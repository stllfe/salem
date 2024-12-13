def search_topk(query: str, k: int = 5) -> list[dict[str, str]]:
  """Get top K search results for the given query (just like in actual browser with a search engine).

  Note: this only returns the URLs and their short descriptions.
    Use `get_page_content` to get the actual page content in a readable format!

  Args:
    query: The search query in a natural form
    k: The amount of pages to return

  Returns:
    A list of K dictionaries with URLs and their descriptions
  """


def get_page_content(url: str) -> str:
  """Extract the readable content from the given page URL in the Markdown format.

  Args:
    url: A string starting with 'http://' or 'https://' or '@' and a number (1..k, both ends included)
      to get the URL from the last `search_topk` call (e.g. '@1' to get the content of the first URL)

  Returns:
    A markdown string with the page content
  """


# TODO: define the requirements
# def get_wiki_page(query: str, k: int = 5) -> str:
