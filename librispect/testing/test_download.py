import librispect as lspct
from librispect.data.download import download_urls, DEV_CLEAN_URL


def test_download():
    download_urls([DEV_CLEAN_URL])
    assert (lspct.paths.RAW_DIR / "dev-clean.tar.gz").is_file()


if __name__ == "__main__":
    test_download()
