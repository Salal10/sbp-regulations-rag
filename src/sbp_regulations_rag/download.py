from pathlib import Path

import httpx

DOCS = {
    "emi_regulations_2023.pdf": "https://archive.sbp.org.pk/psd/2023/C3-Enclosure-Regulations-EMIs.pdf",
    "digital_banks_framework_2022.pdf": "https://archive.sbp.org.pk/bprd/2022/C1-Annex.pdf",
    "branchless_banking_regulations_2019.pdf": "https://archive.sbp.org.pk/bprd/2019/C10-Branchless-Banking-Regulations.pdf",
    "pso_psp_rules_2014.pdf": "https://archive.sbp.org.pk/psd/2014/C3-Annex.pdf",
    "complaint_handling_guidelines_2016.pdf": "https://archive.sbp.org.pk/cpd/2016/C1-Annex.pdf",
    "eft_act_2007.pdf": "https://archive.sbp.org.pk/psd/2007/EFT_Act_2007.pdf",
}
RAW_DIR = Path("data/raw")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in DOCS.items():
        path = RAW_DIR / name
        if path.exists():
            print(f"skip {name} (already downloaded)")
            continue
        response = httpx.get(url, headers=HEADERS, follow_redirects=True, timeout=60)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type:
            raise ValueError(f"{name}: expected a PDF, got {content_type}")
        path.write_bytes(response.content)
        print(f"saved {name} ({len(response.content) // 1024} KB)")


if __name__ == "__main__":
    main()