from tools.core.backend.web import Browser
from tools.runtime import ISO8061_DATE
from tools.runtime import runtime
from tools.types import WebLink
from tools.types import WikiExtract


browser = runtime.get_backend(Browser)


def _format_link(l: WebLink) -> str:
  return f"[{l.uid}] {l.title} [URL]({l.url}):\n\t{l.caption}"


def _format_wiki(w: WikiExtract) -> str:
  header = f"{w.title}/{w.section}" if w.section else w.title
  header = header.strip()
  return (
    f"[{w.uid}] **{header}** [URL]({w.url}):\n\t{w.content}\n\t_last edited @ {w.updated_at.strftime(ISO8061_DATE)}_"
  )


def search_topk(query: str, k: int = 3) -> str:
  """Get top K search results for the given query (just like a desktop browser with a search engine).

  Note: this only returns the URLs and their short descriptions.
    Use `get_page_content` to get the actual page content in a readable format!

  Args:
    query: The search query in a natural form
    k: The amount of top relevant pages to return

  Returns:
    A list of K links with URLs, their descriptions and short unique IDs
    for referencing in answers or other functions
  """

  if links := browser.search_topk(query, k=k):
    return "Search results:\n- " + "- ".join([_format_link(l) + "\n" for l in links])
  return f"No results found for search {query!r}"


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


def search_wiki(query: str, k: int = 3) -> str:
  """Get top K search results from Wikipedia for the given query.

  Hint: use it whenever user asks for factual question like a person, company, city name etc.

  Note: it returns short relevant extracts (not full pages) if found.

  Args:
    query: The search query in a natural form
    k: The amount of top relevant extracts to return

  Returns:
    A list of K short Wikipedia extracts with their full URLs and some metadata.
  """

  if wikis := browser.search_wiki(query, k=k):
    return "Relevant wikipedia extracts:\n- " + "- ".join([_format_wiki(w) + "\n" for w in wikis])
  return f"No results found for search {query!r}"
