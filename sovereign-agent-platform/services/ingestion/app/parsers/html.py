from bs4 import BeautifulSoup


def parse_html(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        soup = BeautifulSoup(handle.read(), "html.parser")
    return soup.get_text("\n")
