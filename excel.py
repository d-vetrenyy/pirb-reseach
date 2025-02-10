import re
import sys
import nltk
import json
import csv
import openpyxl
import requests
import numpy as np
import pandas as pd
import pirb.evals as evals
import pirb.tokenization as tokenization
from bs4 import BeautifulSoup
from dataclasses import dataclass


SENTENCE_TOKENIZE_FIXES = [
    "англ.", "рус."
]


@dataclass
class ScoredSentence:
    score: int
    sentence: str

    @staticmethod
    def pattern() -> str:
        return r"[Ss]core\s*=\s*(\d+)\s*:\s*(.+?)(?=\s*[Ss]core =|\Z)"

    @staticmethod
    def from_blob(blob) -> list:
        blob = str(blob)
        result = []
        for score, sentence_blob in re.findall(ScoredSentence.pattern(), blob):
            for sentence in tokenization.split_by_sentence(sentence_blob, fixes=SENTENCE_TOKENIZE_FIXES):
                result.append(ScoredSentence(score, sentence))
        return result


def content_of_link(link: str) -> bytes | None:
    try:
        response = requests.get(link)
        response.encoding = 'utf-8'
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"ERROR:\tfetching {link}: {e}")
        return None
    except Exception as e:
        print(f"ERROR:\tprocessing {link}: {e}")
        return None


def article_of_content(content: bytes) -> str:
    soup = BeautifulSoup(content, "lxml")
    content_div = soup.find(id="mw-content-text")

    infobox = content_div.find(class_='infobox')
    toc = content_div.find(id="toc")
    figcaptions = content_div.find_all(name="figcaption")
    editsections = content_div.find_all(class_="mw-editsection")

    if infobox != None: infobox.extract()
    if toc != None: toc.extract()
    for figcaption in figcaptions:
        figcaption.extract()
    for editsection in editsections:
        editsection.extract()

    return content_div.get_text()


def get_similar_sentences(lst1: list[str], lst2: list[str], strategy: str | None = None) -> list[tuple[int, int]]:
    strategy = strategy or "pyindex"
    result = []

    if strategy == "pyindex":
        for l1_index, sent in enumerate(lst1):
            try:
                l2_index = lst2.index(sent)
                result.append((l1_index, l2_index))
            except ValueError as e:
                pass
    else:
        raise ValueError("unknown strategy")


if __name__ == "__main__":
    workbook = openpyxl.load_workbook("facts.xlsx")

    facts = {}
    sentences = {}
    rels = []

    for sheetname in workbook.sheetnames:
        sheet = workbook[sheetname]

        for fact_cell in sheet['B'][1:]:
            facts[f"{sheetname}::{fact_cell.row}"] = fact_cell.value

            related_rows = sheet[fact_cell.row][2:]
            link_refblob_pairs = zip(related_rows[::2], related_rows[1::2])
            link_refblob_pairs = map(lambda pair: (pair[0].value, pair[1].value), link_refblob_pairs)

            for link, refblob in link_refblob_pairs:
                if link == None or refblob == None:
                    print(f"WARN:\tskipping empty link/blob pair (in {sheetname}:{fact_cell.row})", file=sys.stderr)
                    continue

                scored_sentences: list[ScoredSentence] = ScoredSentence.from_blob(refblob)
                if len(scored_sentences) == 0:
                    print(f"WARN:\tskipping link with unparsable blob (in {sheetname}:{fact_cell.row})", file=sys.stderr)
                    continue

                print(f"INFO:\tfetching {link}...", file=sys.stderr)
                link_content = content_of_link()
                if link_content == None:
                    continue

                article = article_of_content(link_content)
                article_sentences = tokenization.split_by_sentence(article, fixes=SENTENCE_TOKENIZE_FIXES)
                for i, article_sentence in enumerate(article_sentences):
                    sentences[f"{link}::{i}"] = article_sentence

                similar_sentences_idxs = get_similar_sentences(article_sentences, [el.sentence for el in scored_sentences])
                similarity_recall = evals.recall_value(len(similar_sentences_idxs), len(scored_sentences))
                similar_sentences = [
                    (article_sentences[article_idx], scored_sentences[table_idx].sentence)
                    for article_idx, table_idx in similar_sentences_idxs]

                for article_sentence_idx, table_sentence_idx in similar_sentences_idxs:
                    rels.append((f"{sheetname}::{fact_cell.row}",
                                 f"{link}::{article_sentence_idx}",
                                 scored_sentences[table_sentence_idx].score))
