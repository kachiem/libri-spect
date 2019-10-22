import librispect as lspct
from librispect.data.generate_wav import conversion


def test_generate_wav():
    conversion()
    assert (lspct.paths.WAV_DIR / "1272-128104-0000.wav").is_file()
    assert (lspct.paths.WAV_DIR / "8842-304647-0013.wav").is_file()


if __name__ == "__main__":
    test_generate_wav()
