def parse_html(path: str) -> str:
    from bs4 import BeautifulSoup

    with open(path, "r", encoding="utf-8") as handle:
        soup = BeautifulSoup(handle.read(), "html.parser")
    return soup.get_text("\n")
