from attrs import define
from attrs import field
from cachetools import TTLCache
from cachetools import cachedmethod
from duckduckgo_search import DDGS

from salem.tools.core.backend.web.base import SearchEngine
from salem.tools.types import WebLink


@define
class DuckDuckGoSearch(SearchEngine):
  ddgs: DDGS = field(factory=DDGS)
  cache: TTLCache = field(factory=lambda: TTLCache(maxsize=100, ttl=3600))

  @cachedmethod(lambda self: self.cache)
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
