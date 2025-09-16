# Stanford's WikiChat:
# https://search.genie.stanford.edu/redoc


from attrs import define
from attrs import field
from requests import Session

from salem.tools.core.backend.web.base import SearchEngine
from salem.tools.types import Language
from salem.tools.types import WikiExtract


URL = "https://search.genie.stanford.edu/wikipedia_20250320"
FALLBACK_LANGUAGE: Language = "en"


@define
class WikiChatSearch(SearchEngine[WikiExtract]):
  rerank: bool = True
  score_threshold: float = 0.25
  language: Language = "en"
  session: Session = field(factory=Session)

  def _post(self, query: str, k: int, language: str) -> list[dict]:
    request = {
      "query": [query],
      "rerank": self.rerank,
      "num_blocks": k,
      "num_blocks_to_rerank": min(k**2, 100),
      "search_filters": [
        {"field_name": "language", "filter_type": "eq", "field_value": language}
        # this doesn't work as expected! returns empty results always
        # TODO: make an issue for this on github
        # {"field_name": "similarity_score", "filter_type": "gt", "field_value": self.score_threshold},
      ],
    }

    response = self.session.post(URL, json=request)
    response.raise_for_status()

    if results := response.json():
      return results[0]["results"]

    return []

  def search(self, query: str, k: int = 3) -> list[WikiExtract]:
    # get rescored wiki pages for both languages
    results = self._post(query, k, self.language) + self._post(query, k, FALLBACK_LANGUAGE)

    def extract(r: dict) -> WikiExtract:
      return WikiExtract(
        url=r["url"],
        title=r["document_title"],
        content=r["content"],
        language=r["block_metadata"].get("language", "unknown"),
        updated_at=r["last_edit_date"],
        section=r.get("section_title"),
      )

    results = filter(lambda r: r["similarity_score"] >= self.score_threshold, results)
    results = sorted(results, key=lambda r: r["similarity_score"], reverse=True)

    return list(map(extract, results[:k]))
