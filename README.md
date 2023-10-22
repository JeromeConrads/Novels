# Novels
An analysis of project gutenberg novels

Frok made by Jerome Conrads for Bachelor Thesis

pip install gutenberg

create folder gutenberg
download rdf-files from gutenberg:https://www.gutenberg.org/cache/epub/feeds/

maybe change GUTENBERG_MIRROR variable to working mirror

run process_gutenberg.py

create folder results and texts
run process_all_texts.py
-adapted code to python3
added variable to not recalculate results if already done
-TODO: better way to get a good pool
-TODO: is pooling actually faster?


run post_process.py

cache is not needed presumably
if no cache created: 
pip install --upgrade gutenberg
and maybe clear cache

post_process creates X50-1.csv and y50-1.csv in /data 
changed average by removing a /1000 as that made a division by 0 for books with less then 1000 lines
may lead to problems later

old data used 3403 lines, ERF 0.589
mine 7731 lines, ERD 0.616
