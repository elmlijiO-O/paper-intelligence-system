import requests
import xml.etree.ElementTree as ET


ARXIV_API = "http://export.arxiv.org/api/query"
NS = "{http://www.w3.org/2005/Atom}"


def fetch_papers(query: str, max_results: int = 20) -> list[dict]:
    """
    Query the arXiv API and return a list of Paper objects.
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    response = requests.get(ARXIV_API, params=params, timeout=10)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    papers = []

    for entry in root.findall(f"{NS}entry"):
        raw_id = entry.find(f"{NS}id").text.strip()
        arxiv_id = raw_id.split("/abs/")[-1].replace("v1", "").replace("v2", "").strip()

        title = entry.find(f"{NS}title").text.strip().replace("\n", " ")
        abstract = entry.find(f"{NS}summary").text.strip().replace("\n", " ")

        authors = [
            a.find(f"{NS}name").text.strip()
            for a in entry.findall(f"{NS}author")
        ]

        published = entry.find(f"{NS}published").text.strip()
        year = int(published[:4])

        categories = [
            tag.get("term")
            for tag in entry.findall("{http://arxiv.org/schemas/atom}primary_category")
        ]
        if not categories:
            categories = [
                tag.get("term")
                for tag in entry.findall("{http://www.w3.org/2005/Atom}category")
            ]

        papers.append({
            "id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "year": year,
            "categories": categories,
        })

    return papers


def fetch_paper_by_id(arxiv_id: str) -> dict | None:
    """
    Fetch a single paper by its arXiv ID.
    Returns a Paper object or None if not found.
    """
    params = {
        "id_list": arxiv_id,
        "max_results": 1,
    }
    response = requests.get(ARXIV_API, params=params, timeout=10)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    entries = root.findall(f"{NS}entry")

    if not entries:
        return None

    entry = entries[0]

    if entry.find(f"{NS}title") is None:
        return None

    raw_id = entry.find(f"{NS}id").text.strip()
    resolved_id = raw_id.split("/abs/")[-1].split("v")[0].strip()

    title = entry.find(f"{NS}title").text.strip().replace("\n", " ")
    abstract = entry.find(f"{NS}summary").text.strip().replace("\n", " ")

    authors = [
        a.find(f"{NS}name").text.strip()
        for a in entry.findall(f"{NS}author")
    ]

    published = entry.find(f"{NS}published").text.strip()
    year = int(published[:4])

    categories = [
        tag.get("term")
        for tag in entry.findall("{http://arxiv.org/schemas/atom}primary_category")
    ]

    return {
        "id": resolved_id,
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "url": f"https://arxiv.org/abs/{resolved_id}",
        "year": year,
        "categories": categories,
    }