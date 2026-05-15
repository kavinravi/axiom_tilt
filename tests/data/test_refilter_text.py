"""Tests for the v1 → v2 text refilter.

The filter has three jobs:
  - Keep coherent English-prose runs (10-K business sections, 8-K narratives).
  - Drop XBRL inline-data dumps (`us-gaap:Foo cik date iso4217:USD ...`).
  - Drop residual HTML/CSS attribute fragments (`style="..." <span ...>`).
"""
from __future__ import annotations

from src.data.refilter_text import is_content_token, refilter_text


class TestIsContentToken:
    def test_plain_words(self) -> None:
        assert is_content_token("revenue")
        assert is_content_token("Apple")
        assert is_content_token("well-known")
        assert is_content_token("Apple's")
        assert is_content_token("Inc.")

    def test_rejects_xbrl_refs(self) -> None:
        assert not is_content_token("us-gaap:CoreDepositsMember")
        assert not is_content_token("iso4217:USD")
        assert not is_content_token("xbrli:shares")
        assert not is_content_token("c:OtherCustomerRelationshipsMember")

    def test_rejects_dates_and_numbers(self) -> None:
        assert not is_content_token("2021-03-31")
        assert not is_content_token("1,234,567")
        assert not is_content_token("$1.2B")
        assert not is_content_token("12345")

    def test_rejects_html_attrs(self) -> None:
        assert not is_content_token('style="background-color:red"')
        assert not is_content_token('colspan="3"')
        assert not is_content_token('font-family:Times')  # has ':'
        assert not is_content_token("background-color:#cceeff")

    def test_rejects_urls_and_paths(self) -> None:
        assert not is_content_token("https://example.com")
        assert not is_content_function_safe("/path/to/thing")

    def test_rejects_pure_garbage(self) -> None:
        assert not is_content_token("&K^*C")
        assert not is_content_token("(-1+B")
        assert not is_content_token("a")  # too short
        assert not is_content_token("x" * 50)  # too long


def is_content_function_safe(x: str) -> bool:
    """Tiny helper so the missing-token check above doesn't typo-trip."""
    return is_content_token(x)


class TestRefilterText:
    def test_keeps_prose_passage(self) -> None:
        prose = (
            "Citigroup Inc. reported revenue of seventeen billion dollars in the first "
            "quarter, reflecting strong consumer banking performance across all major "
            "geographies. Management noted that credit quality remained stable while "
            "operating expenses increased modestly due to investment in technology and "
            "regulatory compliance. The company expects continued growth in fee-based "
            "businesses over the next several quarters as macroeconomic conditions "
            "improve and corporate clients resume capital markets activity at normal "
            "historical levels."
        )
        out = refilter_text(prose)
        assert "Citigroup" in out
        assert "revenue" in out
        assert "regulatory" in out
        # Most of the prose should survive
        assert len(out) > 0.7 * len(prose)

    def test_drops_pure_xbrl_dump(self) -> None:
        xbrl = " ".join(
            [
                "0000831001",
                "us-gaap:CoreDepositsMember",
                "2021-03-31",
                "iso4217:USD",
                "0000831001",
                "us-gaap:CashAndCashEquivalentsMember",
                "2021-03-31",
                "xbrli:shares",
                "0000831001",
                "dei:EntityCommonStockSharesOutstanding",
                "2021-03-31",
            ]
            * 50  # 550 tokens of pure XBRL
        )
        out = refilter_text(xbrl)
        # The entire dump should be dropped (or reduced to noise-free fragment)
        assert "us-gaap" not in out
        assert "iso4217" not in out
        # Most of input is gone
        assert len(out) < 0.1 * len(xbrl)

    def test_drops_residual_html_attrs(self) -> None:
        html = (
            '<td colspan="3" style="background-color:#cceeff;padding:2px;'
            'text-align:right">Total assets</td><td style="border-top:1pt '
            'solid #000000;text-align:right">123,456</td>' * 30
        )
        out = refilter_text(html)
        # BeautifulSoup will recover "Total assets 123,456 ..." — that gets
        # filtered by the density check (mostly numbers + short fragments)
        assert "style=" not in out
        assert "background-color" not in out
        assert "colspan" not in out

    def test_mixed_prose_then_xbrl_keeps_only_prose(self) -> None:
        prose = (
            "The Company is a leading global financial services holding company "
            "providing banking insurance and capital markets services to consumers "
            "and corporate clients around the world. Our principal operating "
            "subsidiaries include Citibank and Citigroup Global Markets which "
            "together generate the majority of consolidated revenue. Performance "
            "in the most recent quarter reflected continued strength in both "
            "consumer and institutional segments offset by elevated funding costs. "
        )
        xbrl = " ".join(["us-gaap:Foo", "2021-03-31", "iso4217:USD", "0000831001"] * 200)
        out = refilter_text(prose + " " + xbrl)
        assert "Company" in out
        assert "banking" in out
        assert "us-gaap" not in out
        assert "iso4217" not in out

    def test_empty_text(self) -> None:
        assert refilter_text("") == ""

    def test_short_prose_kept(self) -> None:
        short = "This is a brief filing announcing the appointment of a new director."
        out = refilter_text(short)
        assert "director" in out

    def test_short_garbage_dropped(self) -> None:
        garbage = "1234 5678 us-gaap:Foo iso4217:USD 2021-01-01"
        assert refilter_text(garbage) == ""
