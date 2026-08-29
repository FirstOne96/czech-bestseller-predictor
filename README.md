# Data-Driven Selection of Foreign Books for Czech Translation

Bachelor's thesis project. Predicts which foreign books Czech publishers will
pick for translation, learning from past acquisitions: Goodreads works matched
to Czech translation records in the National Library catalogue (NKC) versus
comparable foreign works never translated. All features are leakage-free
(computed strictly before each book's decision cutoff).

Everything runs from `notebooks/main.ipynb`. See `CLAUDE.md` for full
documentation (pipeline, matching cascade, label and leakage rules).
