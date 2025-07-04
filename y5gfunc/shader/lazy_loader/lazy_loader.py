"""A lazy loader for remote resources."""

import requests
import time
from typing import Optional, Union, cast


class LazyVariable:
    """
    Represents a variable that is lazily loaded from a URL.

    The content is downloaded on first access and cached for a specified
    time-to-live (TTL). It supports retries with delays on download failure.
    The content can be treated as text or binary.
    """

    def __init__(
        self,
        url: str,
        binary: bool = False,
        timeout: int = 10,
        max_retries: int = 5,
        retry_delay: int = 2,
        cache_ttl: int = 3600,
    ):
        """
        Initializes the LazyVariable instance.

        Args:
            url: The URL to download the content from.
            binary: Whether the content is binary. Defaults to False.
            timeout: The request timeout in seconds. Defaults to 10.
            max_retries: The maximum number of retries. Defaults to 5.
            retry_delay: The delay between retries in seconds. Defaults to 2.
            cache_ttl: The time-to-live for the cache in seconds. Defaults to 3600.
        """
        self.url: str = url
        self.binary: bool = binary
        self.timeout: int = timeout
        self.max_retries: int = max_retries
        self.retry_delay: int = retry_delay
        self.cache_ttl: int = cache_ttl

        self._cached_content: Optional[Union[str, bytes]] = None
        self._cache_timestamp: Optional[float] = None

    def _is_cache_valid(self) -> bool:
        """Checks if the cached content is still valid."""
        return (
            self._cache_timestamp is not None
            and (time.time() - self._cache_timestamp) < self.cache_ttl
        )

    def _download(self) -> Union[str, bytes]:
        """
        Downloads the content from the URL.

        If a valid cached version exists, it is returned.
        Otherwise, it downloads the content, caches it, and then returns it.

        Raises:
            RuntimeError: If the download fails after all retry attempts.

        Returns:
            The downloaded content.
        """
        if self._is_cache_valid():
            return cast(Union[str, bytes], self._cached_content)

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(self.url, timeout=self.timeout)
                response.raise_for_status()

                content: Union[str, bytes]
                if self.binary:
                    content = response.content
                else:
                    content = response.text
                self._cached_content = content
                self._cache_timestamp = time.time()

                return content

            except requests.RequestException as e:
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                else:
                    raise RuntimeError(
                        f"Failed to download {self.url} after {self.max_retries} attempts: {e}"
                    )
        # This part should be unreachable, but it makes type checkers happy.
        raise RuntimeError(f"Failed to download {self.url}")

    def __str__(self) -> str:
        """
        Returns the string representation of the content.

        If the content is binary, it will be decoded.
        """
        content = self._download()
        if self.binary:
            return cast(bytes, content).decode()
        return cast(str, content)

    def __bytes__(self) -> bytes:
        """
        Returns the bytes representation of the content.

        If the content is text, it will be encoded.
        """
        content = self._download()
        if self.binary:
            return cast(bytes, content)
        return cast(str, content).encode()

    def refresh(self) -> None:
        """Forces a refresh of the content by clearing the cache."""
        self._cached_content = None
        self._cache_timestamp = None
