"""Open the self-contained, offline browser showcase."""

from pathlib import Path
import webbrowser


PAGE = Path(__file__).with_name("index.html")


def main() -> None:
    url = PAGE.resolve().as_uri()
    print(f"Opening Dyna Putt from {url}")
    if not webbrowser.open(url):
        print("Open dynaputt/index.html in your browser to play.")


if __name__ == "__main__":
    main()
