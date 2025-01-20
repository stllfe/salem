from attrs import define
from attrs import field
from duckduckgo_search import DDGS

from tools.core.backend.web.base import SearchEngine
from tools.types import WebLink


@define
class DuckDuckGoSearch(SearchEngine):
  ddgs: DDGS = field(factory=DDGS)

  def search(self, query: str, k: int = 3) -> list[WebLink]:
    if results := self.ddgs.text(query, max_results=k):

      def weblink(r: dict) -> WebLink:
        return WebLink(
          url=r["href"],
          title=r.get("title"),
          caption=r.get("body"),
        )

      return list(map(weblink, results))
    return results
