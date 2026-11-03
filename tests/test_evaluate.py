from types import SimpleNamespace

import pandas as pd
import pytest

from sbp_regulations_rag.evaluate import first_hit_rank, is_hit


def make(doc, page_start, page_end):
    return SimpleNamespace(doc=doc, page_start=page_start, page_end=page_end)


def test_same_doc_same_page_is_a_hit():
    chunk = make("eft_act_2007", 9, 9)
    question = make("eft_act_2007", 9, 9)

    result = is_hit(chunk, question)

    assert result is True


def test_different_doc_is_not_a_hit():
    chunk = make("emi_regulations_2023", 9, 9)
    question = make("eft_act_2007", 9, 9)

    assert is_hit(chunk, question) is False