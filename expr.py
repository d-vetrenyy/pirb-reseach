import numpy as np
import pandas as pd
import re
import sys
import openpyxl
import requests
from dataclasses import dataclass
from bs4 import BeautifulSoup
import pirb.evals as evals
import pirb.tokenization as tokenization
import json
import csv


SENTENCE_TOKENIZE_FIXES = [
    "англ.", "рус."
]


@dataclass
class ScoredSentence:
    score: int
    sentence: str

    @staticmethod
    def pattern() -> str:
        return r"[Ss]core\s*=\s*([\.\d]+)\s*:\s*(.+?)(?=\s*[Ss]core =|\Z)"


def scored_sentences_from_blob(blob) -> list[ScoredSentence]:
    blob = str(blob)
    result = []
    for score, sentence_blob in re.findall(ScoredSentence.pattern(), blob):
        for sentence in tokenization.split_by_sentence(sentence_blob, fixes=SENTENCE_TOKENIZE_FIXES):
            result.append(ScoredSentence(float(score), sentence))
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


if __name__ == '__main__':
    workbook = openpyxl.load_workbook('facts.xlsx')

    facts = {}
    sentences = {}
    rels = []

    for sheetname in workbook.sheetnames:
        if sheetname != 'Афанасьева':
            break
        sheet = workbook[sheetname]

        for fact_cell in sheet['B'][1:]:
            facts[f"{sheetname}::{fact_cell.row}"] = fact_cell.value # adding new fact in memory

            related_rows = sheet[fact_cell.row][2:]
            link_refblob_pairs = zip(related_rows[::2], related_rows[1::2])
            link_refblob_pairs = map(lambda pair: (pair[0].value, pair[1].value), link_refblob_pairs)

            for link, refblob in link_refblob_pairs:
                if link == None or refblob == None:
                    print(f"WARN:\tskipping empty link/blob pair (in {sheetname}:{fact_cell.row})", file=sys.stderr)
                    continue

                scored_sentences = scored_sentences_from_blob(refblob)
                if len(scored_sentences) == 0:
                    print(f"WARN:\tskipping link with unparsable blob (in {sheetname}:{fact_cell.row})", file=sys.stderr)
                    continue

                print(f"INFO:\tfetching {link}...", file=sys.stderr)
                link_content = content_of_link(link)
                if link_content == None:
                    continue

                article = article_of_content(link_content)
                article_sentences = tokenization.split_by_sentence(article, fixes=SENTENCE_TOKENIZE_FIXES)

                for i, article_sentence in enumerate(article_sentences):
                    sentences[f"{link}::{i}"] = article_sentence

                df_table = pd.DataFrame({
                    'sentence': [sc.sentence for sc in scored_sentences],
                    'score': [sc.score for sc in scored_sentences],
                })
                df_article = pd.DataFrame(article_sentences, columns=['sentence'])
                merged_df = df_article.merge(df_table, on='sentence', how='left')
                result = merged_df[merged_df['score'].notna()]

                for row in result.itertuples():
                    rels.append((f"{sheetname}::{fact_cell.row}", f"{link}::{row.Index}", row.score))

    print('\n\n\n\n')
    # print(facts)
    # print(sentences)
    # print(rels)
    with open('corpus/facts.json', 'w') as factsfile:
        json.dump(facts, factsfile, ensure_ascii=False, indent=1)
    with open('corpus/sentences.json', 'w') as sentfile:
        json.dump(sentences, sentfile, ensure_ascii=False, indent=1)
    with open('corpus/rels.tsv', 'w') as relsfile:
        csv.writer(relsfile, delimiter='\t').writerows(rels)
