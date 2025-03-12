import re

from abc import ABC
from abc import abstractmethod
from typing import ClassVar, Generic, TypeVar

import requests_cache as rc
import retry_requests as rr

from attrs import define
from attrs import field
from markdownify import markdownify
from requests import Session
from requests.auth import HTTPBasicAuth

from tools.types import Language
from tools.types import WebLink
from tools.types import WikiExtract
from tools.utils import truncate_content


LINK_UID_PREFIX = "@"
DEFAULT_LANGUAGE: Language = "en"
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"  # noqa
DEFAULT_HEADERS = {"User-Agent": USER_AGENT}


cache_session = rc.CachedSession(".cache", expire_after=3600)
retry_session = rr.retry(cache_session, retries=3, backoff_factor=0.2)


T = TypeVar("T")


class SearchEngine(ABC, Generic[T]):
  @abstractmethod
  def search(self, query: str, k: int) -> list[T]:
    pass


def is_url_reference(url: str) -> bool:
  return url.startswith(LINK_UID_PREFIX)


@define
class Browser:
  web: SearchEngine[WebLink]
  wiki: SearchEngine[WikiExtract]
  session: Session = field(default=retry_session)
  state: dict[str, WebLink] = field(factory=dict, init=False)

  HEADERS: ClassVar[dict] = DEFAULT_HEADERS

  def search_topk(self, query: str, k: int = 3) -> list[WebLink]:
    if links := self.web.search(query, k=k):
      self.state.clear()  # only keep the latest search results
      self.state.update(({link.uid: link for link in links}))
    return links

  def search_wiki(self, query: str, k: int = 3) -> list[WikiExtract]:
    return self.wiki.search(query, k=k)

  def get_cached_link(self, url: str) -> WebLink | None:
    return self.state.get(url.lstrip(LINK_UID_PREFIX))

  def resolve_url(self, url: str) -> str:
    if not is_url_reference(url):
      return url
    if not (link := self.get_cached_link(url)):
      raise ValueError(f"No such page found in session: {url}")
    return link.url

  def get_page_content(self, url: str) -> str:
    # get the URL from the cache if it's a link UID
    url = self.resolve_url(url)

    # send a GET request to the URL
    response = self.session.get(url, headers=self.HEADERS)
    if response.status_code in (401, 400):
      response = self.session.get(url, headers=self.HEADERS, auth=HTTPBasicAuth("user", "pass"))

    response.raise_for_status()  # raise an exception for bad status codes

    # convert the HTML content to Markdown
    markdown_content = markdownify(response.text).strip()

    # remove multiple line breaks
    markdown_content = re.sub(r"\n{3,}", "\n", markdown_content)

    # cache visited pages
    content = truncate_content(markdown_content, 10000)
    return content
