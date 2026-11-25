# Data sources

No corpus is redistributed in this repository. Download each from its source; check its licence before reuse. `n` is the number of records scored in the paper (Table I).

## Text corpora (email, SMS, comments)

| Corpus | n | Source |
|---|---|---|
| Enron, SpamAssassin, Ling, CEAS-08, TREC-06, TREC-07 | 29,767 / 5,808 / 2,859 / 39,154 / 16,399 / 53,757 | Curated release of Champa, Rabbi and Zibran (ICMI 2024): Zenodo, doi:10.5281/zenodo.8339691 |
| SMS Spam Collection | 5,574 | UCI Machine Learning Repository, "SMS Spam Collection" (Almeida et al., DocEng 2011) |
| YouTube Spam Collection | 1,956 | UCI Machine Learning Repository, "YouTube Spam Collection" (Alberto et al., ICMLA 2015) |
| Nazario, Nigerian Fraud | phishing only (transfer targets) | Same Zenodo release as above |
| Smishing (IMC 2025) | 22,003 used, of 22,078 English messages | github.com/reportsmishing/Smishing-Dataset-IMC25 (`dataset/final_dataset_output.csv`, `language == "English"`); CC BY 4.0 |
| Synthetic phishing email corpus (50k) | 50,000 | Kaggle: kaggle.com/datasets/mobeenfatimah/phishing-email-detection-dataset-50k-emails |

Excluded: TREC-05 (its ham is Enron mail) and the Mishra and Soni SMS phishing set (Mendeley, doi:10.17632/f45bkkt8pr.1), which credits the SMS Spam Collection.

## URL corpora

| Corpus | n | Source |
|---|---|---|
| Ebbu2017 | 73,575 | github.com/ebubekirbbr/pdd (Sahingoz et al., Expert Syst. Appl. 2019) |
| Tiwari, "Phishing Site URLs" | 549,346 rows (507,195 unique URLs) | Kaggle: kaggle.com/datasets/taruntiwarihp/phishing-site-urls |
| Siddhartha, "Malicious URLs dataset" | 522,214 = benign 428,103 + phishing 94,111 (defacement and malware classes dropped; 651,191 rows in total) | Kaggle: kaggle.com/datasets/sid321axn/malicious-urls-dataset |
| Hannousse and Yahiouche | 11,430 | Mendeley Data (Eng. Appl. Artif. Intell. 2021) |
| PhiUSIIL | 235,795 | UCI Machine Learning Repository, "PhiUSIIL Phishing URL Dataset" (Prasad and Chandra, Comput. Secur. 2024) |
| URL-2025 | 1,061,019 (61,019 phishing, 1,000,000 legitimate; one empty-URL row dropped) | github.com/Laashya16/phishing-url-dataset |
| URL-2026 | 5,000 (2,547 benign, 2,453 malicious) | github.com/khanren/Malicious-URL-Detection-Dataset, file `Datasets/Dataset II.csv` |

## Overlap rule

A corpus enters the predictive test only if it shares no more than 10% of its records with any corpus already admitted (exact match after normalisation), taking candidates in order of release. Measured overlaps: 86–88% of Tiwari in Siddhartha, 36% of Hannousse in Ebbu2017, 19% of PhiUSIIL in URL-2025, 13% of Ebbu2017 in Tiwari.
