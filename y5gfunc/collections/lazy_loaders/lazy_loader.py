import requests
import time
import logging
from typing import Optional

class LazyVariable:
    def __init__(self, url: str, binary: bool = False, 
                timeout: int = 10, max_retries: int = 3, retry_delay: int = 2, cache_ttl: int = 3600):
        self.url = url
        self.binary = binary
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_ttl = cache_ttl

        self._cached_content: Optional[str | bytes] = None
        self._cache_timestamp: Optional[float] = None

    def _is_cache_valid(self) -> bool:
        return self._cache_timestamp and (time.time() - self._cache_timestamp) < self.cache_ttl

    def _download(self) -> str | bytes:
        if self._is_cache_valid():
            return self._cached_content

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(self.url, timeout=self.timeout)
                response.raise_for_status()

                content = response.content if self.binary else response.text
                self._cached_content = content
                self._cache_timestamp = time.time()

                return content

            except requests.RequestException as e:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(f"Failed to download {self.url} after {self.max_retries} attempts: {e}")

    def __str__(self) -> str:
        content = self._download()
        return content.decode() if self.binary else content

    def __bytes__(self) -> bytes:
        content = self._download()
        return content if self.binary else content.encode()

    def refresh(self) -> None:
        self._cached_content = None
        self._cache_timestamp = None
