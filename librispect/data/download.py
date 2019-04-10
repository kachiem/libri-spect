"""
Downloads LibriVox Dataset to /data/raw
"""

import urllib.request
import librispect as lspct

TRAIN_CLEAN_100_URL = "http://www.openslr.org/resources/12/train-clean-100.tar.gz"
TRAIN_CLEAN_360_URL = "http://www.openslr.org/resources/12/train-clean-360.tar.gz"
TRAIN_OTHER_500_URL = "http://www.openslr.org/resources/12/train-other-500.tar.gz"

DEV_CLEAN_URL = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
DEV_OTHER_URL = "http://www.openslr.org/resources/12/dev-other.tar.gz"

TEST_CLEAN_URL = "http://www.openslr.org/resources/12/test-clean.tar.gz"
TEST_OTHER_URL = "http://www.openslr.org/resources/12/test-other.tar.gz"

lspct.paths.RAW_DIR.mkdir(parents=True, exist_ok=True)

URLS = [
    TRAIN_CLEAN_100_URL,
    TRAIN_CLEAN_360_URL,
    TRAIN_OTHER_500_URL,
    DEV_CLEAN_URL,
    DEV_OTHER_URL,
    TEST_CLEAN_URL,
    TEST_OTHER_URL,
]


def download_urls(urls):
    for url in urls:
        print("Downloading %s" % (url))
        urllib.request.urlretrieve(url, lspct.paths.RAW_DIR / url.split("/")[-1])
        print("Completed")


if __name__ == "__main__":
    download_urls(URLS)
